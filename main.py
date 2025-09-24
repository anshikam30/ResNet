from cnn import main as main9
from ResNet import main as main10
from PlainNet import main as main11

if __name__ == '__main__':
    import sys

    version = 11
    if len(sys.argv) > 1:
        version = int(sys.argv[1])

    if version == 9:
        main9()
    if version == 10:
        main10()
    if version == 11:
        main11()
