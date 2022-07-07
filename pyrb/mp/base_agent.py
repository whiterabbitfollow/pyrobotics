import pyrb


class MotionPlanningAgent(pyrb.robot.Manipulator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.goal_state = None

    def set_goal_state(self, goal_state):
        self.goal_state = goal_state


class MotionPlanningAgentActuated(MotionPlanningAgent):

    def __init__(self, robot_data, max_actuation):
        super().__init__(robot_data)
        self.max_actuation = max_actuation
