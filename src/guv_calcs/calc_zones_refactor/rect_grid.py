from axis import Axis1D
from dataclass import dataclass, replace

@dataclass(frozen=True, slots=True)
class RectGrid:
    mins: tuple
    maxs: tuple
    n_pts: tuple | None = None
    spacings: tuple | None = None
    offset: bool = True
    
    def __post_init__(self):
        
        if len(self.mins)!=len(self.maxs):
            raise ValueError(f"number of min ({len(self.mins)}) and max ({len(self.maxs)}) dimensions do not match")
        
        if self.n_pts is not None:
            if len(self.n_pts)!=len(self.mins):
                raise ValueError("n_pts dimensions do not match min/max dimensions")
        if self.spacings is not None:
            if len(self.spacings)!=len(self.mins):
                raise ValueError("spacing dimensions do not match min/max dimensions")

        if len(self.mins) > 3:
            raise ValueError("Maximum of three dimensions allowed")
            
        if len(self.mins) < 1:
            raise ValueError("Minimum of one dimension required")

    @property
    def axes(self):        
        axes = [] 
        if self.spacings is None:
            spacings = (None,) * len(self.mins)
        if self.n_pts is None:
            num_points = (None,) * len(self.mins)
        for lo, hi, spacing, n_pts in zip(self.mins, self.maxs, spacings, num_points):
            axis = Axis1D(lo=lo, hi=hi, spacing_init=spacing, n_pts_init=n_pts, offset=self.offset)
            axes.append(axis)            
        return axes
        
    @property
    def points(self):
        return [axis.points for axis in self.axes]
        
    @property
    def num_points(self):
        return np.array([len(pt) for pt in self.points])

    def update(self, **changes):
        return replace(self, **changes)
        
    @property
    def num_x(self):
        if len(num_points)<1:
            return None
        return num_points[0]
        
    @property
    def num_y(self):
        if len(num_points)<2:
            return None
        return num_points[1]
        
    @property
    def num_z(self):
        if len(num_points)<3:
            return None
        return num_points[2]
        
    @property
    def x_spacing(self):
        if len(axes)<1:
            return None
        return self.axes[0].spacing
        
    @property
    def y_spacing(self):
        if len(axes)<2:
            return None
        return self.axes[1].spacing
        
    @property
    def z_spacing(self):
        if len(axes)<3:
            return None
        return self.axes[2].spacing
        
    @property
    def dimensions(self):
        return tuple((axis.lo, axis.hi) for axis in self.axes)

@dataclass(frozen=True, slots=True)
class VolGrid(RectGrid):
    
    @property
    def coords(self):
        X, Y, Z = [
            grid.reshape(-1) for grid in np.meshgrid(*self.points, indexing="ij")
        ]
        coords = np.array((X, Y, Z)).T
        
        return np.unique(coords, axis=0)      
        
@dataclass(frozen=True, slots=True)
class PlaneGrid(RectGrid):
    height: float = 0 
    ref_surface: str = "xy"
    
    def __post_init__(self):
        
        if len(mins) > 2:
            raise ValueError("Too many dimensions for a plane")
    
    @property
    def coords(self):
        X, Y = [grid.reshape(-1) for grid in np.meshgrid(*self.points, indexing="ij")]

        if self.ref_surface.lower() in ["xy"]:
            Z = np.full(X.shape, self.height)
        elif self.ref_surface.lower() in ["xz"]:
            Z = Y
            Y = np.full(Y.shape, self.height)
        elif self.ref_surface.lower() in ["yz"]:
            Z = Y
            Y = X
            X = np.full(X.shape, self.height)
        
        return np.stack([X, Y, Z], axis=-1)
        
    