from examples.world import StaticBoxesWorld
import matplotlib


matplotlib.rc("font", size=16)

world = StaticBoxesWorld()
world.reset()
world.view()

