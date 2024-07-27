"""
    N agents, N landmarks, and obstacles forming a maze.
    Agents are rewarded based on how far any agent is from each landmark.
    Agents are penalized if they collide with other agents or obstacles (maze walls).
    So, agents have to learn to cover all the landmarks while avoiding collisions.
"""
from typing import Optional, Tuple, List
import argparse
import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.getcwd()))

from multiagent.core import World, Agent, Landmark, Obstacle, Wall
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, args: argparse.Namespace) -> World:
        world = World()
        world.world_length = args.episode_length
        world.current_time_step = 0
        # set any world properties first
        world.dim_c = 2
        self.num_agents = args.num_agents
        self.max_speed = args.max_speed
        self.min_dist_thresh = args.min_dist_thresh
        num_landmarks = args.num_agents
        self.num_obstacles = args.num_obstacles
        world.collaborative = args.collaborative
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.id = i
            agent.name = f"agent {i}"
            agent.collide = True
            agent.silent = True
            # NOTE not changing size of agent because of some edge cases;
            # TODO have to change this later
            agent.size = 0.15
            agent.max_speed = self.max_speed
        # add landmarks (goals)
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.id = i
            landmark.name = f"landmark {i}"
            landmark.collide = False
            landmark.movable = False
        #print(world.landmarks)
        # add obstacles (maze walls)
    # add obstacles
        world.obstacles = [Landmark() for i in range(self.num_obstacles)]
        for i, obstacle in enumerate(world.obstacles):
            obstacle.name = f"obstacle {i}"
            obstacle.collide = True
            obstacle.movable = False
        # add walls to form a maze
        world.walls = self.build_maze()
        #self.max_speed = args.max_speed
        # make initial conditions
        self.reset_world(world)
        return world

    def build_maze(self) -> List[Wall]:
        walls = []
        # Define the maze structure with walls
        # Example: a simple maze 
        walls.append(Wall(orient="H", axis_pos=0.5, endpoints=(-1, 1), width=0.1))
        walls.append(Wall(orient="V", axis_pos=-0.5, endpoints=(-1, 1), width=0.1))
        walls.append(Wall(orient="H", axis_pos=-0.5, endpoints=(-1, 1), width=0.1))
        walls.append(Wall(orient="V", axis_pos=0.5, endpoints=(-1, 1), width=0.1))

        walls.append(Wall(orient="H", axis_pos=0.75, endpoints=(-0.5, 0.5), width=0.1))
        walls.append(Wall(orient="V", axis_pos=-0.75, endpoints=(-0.5, 0.5), width=0.1))
        walls.append(Wall(orient="H", axis_pos=0.25, endpoints=(-0.75, 0.75), width=0.1))
        walls.append(Wall(orient="V", axis_pos=-0.25, endpoints=(-0.75, 0.75), width=0.1))
        return walls

    def reset_world(self, world: World) -> None:
        world.current_time_step = 0
        # to track time required to reach goal
        world.times_required = -1 * np.ones(self.num_agents)
        # track distance left to the goal
        world.dist_left_to_goal = -1 * np.ones(self.num_agents)
        # number of times agents collide with stuff
        world.num_obstacle_collisions = np.zeros(self.num_agents)
        world.num_agent_collisions = np.zeros(self.num_agents)
        world.num_wall_collisions = np.zeros(self.num_agents)
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.15, 0.85, 0.15])
        # random properties for obstacles
        for i, obstacle in enumerate(world.obstacles):
            obstacle.color = np.array([0.85, 0.15, 0.15])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        for obstacle in world.obstacles:
            obstacle.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
            obstacle.state.p_vel = np.zeros(world.dim_p)

                # set agents at random positions not colliding with obstacles
        num_agents_added = 0
        while True:
            if num_agents_added == self.num_agents:
                break
            random_pos = np.random.uniform(-1, +1, world.dim_p)
            agent_size = world.agents[num_agents_added].size
            if not self.is_obstacle_collision(random_pos, agent_size, world):
                world.agents[num_agents_added].state.p_pos = random_pos
                world.agents[num_agents_added].state.p_vel = np.zeros(world.dim_p)
                world.agents[num_agents_added].state.c = np.zeros(world.dim_c)
                num_agents_added += 1
        # set landmarks at random positions not colliding with obstacles and
        # also check collisions with placed goals
        num_goals_added = 0
        goals_added = []
        while True:
            if num_goals_added == self.num_agents:
                break
            random_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
            goal_size = world.landmarks[num_goals_added].size
            obs_collision = self.is_obstacle_collision(random_pos, goal_size, world)
            landmark_collision = self.is_landmark_collision(
                random_pos, goal_size, world.landmarks[:num_goals_added]
            )
            if not landmark_collision and not obs_collision:
                world.landmarks[num_goals_added].state.p_pos = random_pos
                world.landmarks[num_goals_added].state.p_vel = np.zeros(world.dim_p)
                num_goals_added += 1
        #####################################################

        ############ find minimum times to goals ############
        if self.max_speed is not None:
            for agent in world.agents:
                self.min_time(agent, world)

     # check collision of entity with obstacles
    def is_obstacle_collision(self, pos, entity_size: float, world: World) -> bool:
        # pos is entity position "entity.state.p_pos"
        collision = False
        for obstacle in world.obstacles:
            delta_pos = obstacle.state.p_pos - pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = obstacle.size + entity_size
            if dist < dist_min:
                collision = True
                break
        return collision

    def is_landmark_collision(self, pos, size: float, landmark_list: List) -> bool:
        collision = False
        for landmark in landmark_list:
            delta_pos = landmark.state.p_pos - pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = size + landmark.size
            if dist < dist_min:
                collision = True
                break
        return collision

        # get min time required to reach to goal without obstacles
    def min_time(self, agent: Agent, world: World) -> float:
        assert agent.max_speed is not None, "Agent needs to have a max_speed"
        agent_id = agent.id
        # get the goal associated to this agent
        landmark = world.get_entity(entity_type="landmark", id=agent_id)
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - landmark.state.p_pos)))
        min_time = dist / agent.max_speed
        agent.goal_min_time = min_time
        return min_time


    def benchmark_data(self, agent: Agent, world: World) -> Tuple:
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                for a in world.agents
            ]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
            for o in world.obstacles:
                if self.is_collision(agent, o):
                    rew -= 1
                    collisions += 1
            for w in world.walls:
                if self.is_collision(agent, w):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1: Agent, entity) -> bool:
        if isinstance(entity, Wall):
            if entity.orient == "H":
                entity_pos = np.array([agent1.state.p_pos[0], entity.axis_pos])
                if entity.endpoints[0] <= agent1.state.p_pos[0] <= entity.endpoints[1]:
                    delta_pos = agent1.state.p_pos[1] - entity_pos[1]
                    dist = np.abs(delta_pos)
                    dist_min = agent1.size + entity.width / 2
                    return dist < dist_min
            else:
                entity_pos = np.array([entity.axis_pos, agent1.state.p_pos[1]])
                if entity.endpoints[0] <= agent1.state.p_pos[1] <= entity.endpoints[1]:
                    delta_pos = agent1.state.p_pos[0] - entity_pos[0]
                    dist = np.abs(delta_pos)
                    dist_min = agent1.size + entity.width / 2
                    return dist < dist_min
        else:
            entity_pos = entity.state.p_pos
            delta_pos = agent1.state.p_pos - entity_pos
            dist = np.sqrt(np.sum(np.square(delta_pos)))
            dist_min = agent1.size + entity.size
            if dist < dist_min:
                return True
            else:
                return False
        

    def reward(self, agent: Agent, world: World) -> float:
        # Agents are rewarded based on minimum agent distance to each landmark,
        # penalized for collisions with other agents and obstacles
        rew = 0
        for l in world.landmarks:
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
            for o in world.obstacles:
                if self.is_collision(agent, o):
                    rew -= 1
            for w in world.walls:
                if self.is_collision(agent, w):
                    rew -= 1
        return rew

    def observation(self, agent: Agent, world: World) -> np.ndarray:
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks + world.obstacles + world.walls:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks + world.obstacles + world.walls:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm
        )

    def info_callback(self, agent: Agent, world: World) -> Tuple:
        # TODO modify this
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        goal = world.get_entity("landmark", agent.id)
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - goal.state.p_pos)))
        world.dist_left_to_goal[agent.id] = dist
        # only update times_required for the first time it reaches the goal
        if dist < self.min_dist_thresh and (world.times_required[agent.id] == -1):
            world.times_required[agent.id] = world.current_time_step * world.dt

        if agent.collide:
            if self.is_obstacle_collision(agent.state.p_pos, agent.size, world):
                world.num_obstacle_collisions[agent.id] += 1
            for a in world.agents:
                if a is agent:
                    continue
                if self.is_collision(agent, a):
                    world.num_agent_collisions[agent.id] += 1
            for w in world.walls:
                if self.is_collision(agent, w):
                    world.num_wall_collisions[agent.id] += 1

        agent_info = {
            "Dist_to_goal": world.dist_left_to_goal[agent.id],
            "Time_req_to_goal": world.times_required[agent.id],
            # NOTE: total agent collisions is half since we are double counting
            "Num_agent_collisions": world.num_agent_collisions[agent.id],
            "Num_obst_collisions": world.num_obstacle_collisions[agent.id],
            "Num_wall_collisions": world.num_wall_collisions[agent.id],
            "Agent_speed": np.linalg.norm(agent.state.p_vel)
        }
        if self.max_speed is not None:
            agent_info["Min_time_to_goal"] = agent.goal_min_time
        return agent_info

    def done(self, agent: Agent, world: World) -> bool:
        # done is False if done_callback is not passed to
        # environment.MultiAgentEnv
        # This is same as original version
        # Check `_get_done()` in environment.MultiAgentEnv
        return False


if __name__ == "__main__":
    from multiagent.environment import MultiAgentOrigEnv
    from multiagent.policy import InteractivePolicy

    # makeshift argparser
    class Args:
        def __init__(self):
            self.num_agents: int = 16
            self.num_obstacles: int = 16
            self.collaborative: bool = False
            self.max_speed: float = 2.0
            self.collision_rew: float = 5
            self.goal_rew: float = 5
            self.min_dist_thresh: float = 0.1
            self.use_dones: bool = False
            self.episode_length: int = 25

    args = Args()

    scenario = Scenario()

    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = MultiAgentOrigEnv(
        world=world,
        reset_callback=scenario.reset_world,
        reward_callback=scenario.reward,
        observation_callback=scenario.observation,
        info_callback=scenario.info_callback,
        done_callback=scenario.done,
        shared_viewer=False,
    )
    # render call to create viewer window
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env, i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    stp = 0
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # step environment
        obs_n, reward_n, done_n, info_n = env.step(act_n)
        # render all agent views
        env.render()
        stp += 1
