import sys
import network

sys.setrecursionlimit(500000)  # we sometimes run into Python's default limit of 999. Note: this can cause a crash!

def main():
    network_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        net, freq = network.load_network(network_file)
        print " Recompiling and saving prediction function."
        network.save_prediction_function(net, output_file, freq)
    except ValueError:
        print " Error loading network, trying prediction function instead."

if __name__ == "__main__":
    main()
