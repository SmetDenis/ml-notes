
def nf(number, decimals=6, dec_point='.', thousands_sep=','):
    num_parts = f"{number:.{decimals}f}".split(".")
    num_parts[0] = f"{int(num_parts[0]):,}".replace(",", thousands_sep)
    return dec_point.join(num_parts) if decimals > 0 else num_parts[0]
