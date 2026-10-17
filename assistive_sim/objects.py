"""Scene objects for the bimanual assistive environment."""
import os

from robosuite.models.objects import BoxObject, CylinderObject, MujocoXMLObject

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


class ForkObject(MujocoXMLObject):
    """Rigid fork (slim handle + rubber grip sleeve, 4 tines). Important sites: 'tip' (tine centre)
    and 'handle_center' (= body origin, centre of the grip sleeve)."""

    REST_HALF_HEIGHT = 0.004  # body origin height above the table when lying flat

    def __init__(self, name="fork"):
        super().__init__(os.path.join(ASSETS, "fork.xml"), name=name,
                         joints=[dict(type="free", damping="0.0005")], obj_type="all",
                         duplicate_collision_geoms=True)

    @property
    def important_sites(self):
        return {**super().important_sites, "tip": self.naming_prefix + "tip",
                "handle": self.naming_prefix + "handle_center"}


class JarObject(MujocoXMLObject):
    """Hollow jar (open-top ring-of-boxes wall + bottom disc) with a knurled lid on a hinge
    joint ('<name>_lid_hinge', z axis, frictionloss + damping)."""

    BOTTOM_OFFSET = 0.045  # body origin height above the jar bottom

    def __init__(self, name="jar"):
        super().__init__(os.path.join(ASSETS, "jar.xml"), name=name,
                         joints=[dict(type="free", damping="0.0005")], obj_type="all",
                         duplicate_collision_geoms=True)

    @property
    def lid_joint(self):
        return self.naming_prefix + "lid_hinge"

    @property
    def lid_body(self):
        return self.naming_prefix + "lid"


def make_plate(name="plate"):
    # flat ceramic disc, 18 cm diameter, 1.2 cm thick
    return CylinderObject(name=name, size=[0.09, 0.006], rgba=[0.95, 0.95, 0.92, 1],
                          density=1500, friction=[1.0, 0.01, 0.001])


def make_food(i):
    """Food piece i: ~1.2-1.4 cm cubes / cylinder (rigid placeholders)."""
    color = [[0.95, 0.75, 0.2, 1], [0.4, 0.8, 0.3, 1], [0.85, 0.45, 0.2, 1]][i % 3]
    if i == 2:
        return CylinderObject(name=f"food{i}", size=[0.007, 0.007], rgba=color, density=900,
                              friction=[1.0, 0.01, 0.001])
    return BoxObject(name=f"food{i}", size=[0.006 + 0.001 * i] * 3, rgba=color, density=900,
                     friction=[1.0, 0.01, 0.001])
