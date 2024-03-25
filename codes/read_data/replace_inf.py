def replace_inf(value):
    import math
    try:
        if math.isinf(value):
            if value > 0:
                return 9999.9999
            else:
                return -9999.9999
        else:
            return value
    except:
        return value