


class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    OKCYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    STRIKE = '\033[9m'

def colorize(text, color):
    return f"{color}{text}{bcolors.ENDC}"
def blue(text):
    return colorize(text, bcolors.BLUE)
def cyan(text):
    return colorize(text, bcolors.OKCYAN)
def red(text):
    return colorize(text, bcolors.RED)
def magenta(text):
    return colorize(text, bcolors.MAGENTA)
def green(text):
    return colorize(text, bcolors.GREEN)
def yellow(text):
    return colorize(text, bcolors.YELLOW)

def underline(text):
    return colorize(text, bcolors.UNDERLINE)
def bold(text):
    return colorize(text, bcolors.BOLD)
def strikethrough(text):
    return colorize(text, bcolors.STRIKE)

def strike(text):
    result = ''
    for c in text:
        result = result + c + '\u0336'
    return result


def make_colors():
    # make colors
    colors = [[255,0,0], [0,0,255], [0,255,255], [255,0,255], [205,255,101]]
    for a in range(55, 255, 101):
        for b in range(0, 255, 101):
            for c in range(0, 255, 175):
                colors.append([a,b,c])
                # aa = '#%02x%02x%02x' % (a,b,c)
                # print (aa)
    hex_colors = ['#%02x%02x%02x' % (x[0],x[1],x[2]) for x in colors]
    return colors, hex_colors



