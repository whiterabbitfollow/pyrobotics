

class MotionPlanningWorld:

    def __init__(self, robot, obstacles, render_engine):
        self.robot = robot
        self.obstacles = obstacles
        self.render_engine = render_engine
        self.collision_manager = None

    def render(self):
        self.render_engine.render(self.robot)