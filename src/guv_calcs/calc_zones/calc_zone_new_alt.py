import numpy as np

def calc_surface(wall, room_dims, height, offset, row_spacing, column_spacing, view_normal=None, **zone_args):
    
    # stub--to be filled in later...
    if wall.lower() in ['north','south']:
        ref_surface = 'xz'
        x,y = room_dims.x, room_dims.z
    elif wall.lower() in ['west','east']:
        ref_surface = 'yz'
        x,y = room_dims.y, room_dims.z
    elif wall.lower() in ['floor','ceiling']:
        ref_surface = 'xy'
        x,y = room_dims.x, room_dims.y
    geometry = RectGrid(x1=0, x2=x, y1=0, y2=y, x_spacing=column_spacing, y_spacing=row_spacing, offset=offset)
    
    return CalcPlane(geometry=geometry, view_normal=view_normal, **zone_args) # encapsulate the vert/horiz stuff with a CalcType preset

class CalcPlane(CalcZone)

    def __init__(geometry, view_normal, vert, horiz, fov_vert, fov_horiz, **zone_args):
        
        super.__init__(**zone_args)
        
        self.geometry = geometry
        self.view_normal = view_normal
        self.vert = vert
        self.horiz = horiz
        self.fov_vert = fov_vert
        self.fov_horiz = fov_horiz
        
        #stub...more to go here
        
                
class CalcZone:
    def __init__(
        zone_id: str | None = None, 
        name: str | None    = None, 
        dose: bool          = False,
        hours: int | float  = 8,
        enabled: bool       = True,
        show_values: bool   = True
        ):
        
        self.zone_id = zone_id
        self.name = name
        self.dose = dose
        self.hours = hours
        self.enabled = enabled
        self.show_values = show_values
        
        self.units = "mJ/cm²" if self.dose else "uW/cm²"
        self.calculator = LightingCalculator(self)
        self.results = ZoneResults()
        
    @property
    def values(self):
        return self._return_val(self.results.values)
        
    @property
    def reflected_values(self):
        return self._return_val(self.results.reflected_values)
        
    @property
    def base_values(self):
        return self._return_val(self.results.base_values)
        
    def _return_val(self)
        if val is None:
            return None
        if self.dose:
            return val * 3.6 * self.hours
        return val
        
    def set_value_type(self, dose):
        self.dose = dose
        self.units = "mJ/cm²" if self.dose else "uW/cm²"
        
    def set_dose_time(self, hours):
        if not isinstance(hours, (float, int):
            raise TypeError("Hours must be numeric")
        self.hours = hours
        
        
## to be filled in later, ignore
@dataclass
class ZoneResults:
    lamp_values_base:       dict[str, np.ndarray] = field(default_factory=dict)
    base_values:  np.ndarray | None = None
    lamp_values:       dict[str, np.ndarray] = field(default_factory=dict)
    reflected_values: np.ndarray | None = None
    values:            np.ndarray | None = None  
    timestamp:         float | None = None
    
class LightingCalculator:
    """
    Performs all computations for a calculation zone
    """

    def __init__(self, zone):
        self.zone = zone
