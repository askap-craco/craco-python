
# define __all__ so someone can import * from steps
# but also as we use this list to work out which modules might have a get_parser() to call

__all__ = ['cluster',
           'time_space_filter',
           'catalog_cross_match',
           'check_filterbanks',
           'check_visibilities']


# actually improt these into the module so we can use them - I'm really not sure whether I'm
# driving python correclty here
from . import *

