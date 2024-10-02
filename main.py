# Some standard imports
import sys
import os
import argparse
from tqdm import tqdm

# Add the directory containing utils.py to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'estimators'))

from utils import *
from estimate import *
from onnx_parser import *

if __name__ == '__main__':

    # Define the accepted arguments
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model",  type = str, help = "Load the desired NN algorithm in the ONNX format")
    parser.add_argument("-c", "--config", type = str, help = "Load the hardware configuration")
    
    args = parser.parse_args()

    print("*********************************************************************************")
    print(f" ******* Model: {args.model}, Hardware_config: {args.config} **********")
    print("*********************************************************************************")

    print("*********************************************************************************")
    print("************************** Load and parse the NN model **************************")
    print("*********************************************************************************")
    layers = load_and_parse(args.model)

    # Visualize the results
    for index in range(len(layers)):
        print(layers[index])

    print("*********************************************************************************")
    print("************************** Estimate the inference time **************************")
    print("*********************************************************************************")

    # Get the processor configuration
    processor_config = User_Input()

    # Check the target hardware configuration
    if args.config == "ARDUINO":
        processor_config.set_parameters(pipeline_en = "Enabled",
                                    pipeline_stages = 5,
                                    lu_en = "Enabled",
                                    lu_factor = 3,
                                    data_forwarding = "Enabled",
                                    memory_latency = 3,
                                    addition = 2,
                                    multiplication = 1,
                                    division = 1,
                                    nr_adders = 0,
                                    nr_multipliers = 0,
                                    SIMD_status = "Disabled",
                                    VLIW_status = "Disabled")
    # Default: Ibex core
    else:
        processor_config.set_parameters(pipeline_en = "Enabled",
                                    pipeline_stages = 5,
                                    lu_en = "Enabled",
                                    lu_factor = 3,
                                    data_forwarding = "Enabled",
                                    memory_latency = 3,
                                    addition = 2,
                                    multiplication = 1,
                                    division = 1,
                                    nr_adders = 0,
                                    nr_multipliers = 0,
                                    SIMD_status = "Disabled",
                                    VLIW_status = "Disabled")
    
    # Initialize the network clock cycles  
    network_cc = 0

    # Calculate the number of clock cycles for each layer
    for index in range(len(layers)):
        if layers[index].operation_type == 'Conv':
            layer_cc = calculate_conv_cc_2(layers[index], processor_config)
            print(f"Clock cycles for layer {layers[index].name} is: {layer_cc}")
        elif layers[index].operation_type == 'ReLU':
            layer_cc = calculate_relu_cc(layers[index], processor_config, 5)
            print(f"Clock cycles for layer {layers[index].name} is: {layer_cc}")
        elif layers[index].operation_type == 'MaxPool':
            layer_cc = calculate_max_pool_cc(layers[index], processor_config, 5)
            print(f"Clock cycles for layer {layers[index].name} is: {layer_cc}")
        elif layers[index].operation_type == 'AvgPool':
            layer_cc = calculate_avg_pool_cc(layers[index], processor_config, 5)
            print(f"Clock cycles for layer {layers[index].name} is: {layer_cc}")
        network_cc += layer_cc
    
    print(f"The clock cycles required for the entire network is: {network_cc}")

    print("*********************************************************************************")
    print("************************** Estimate the area **************************")
    print("*********************************************************************************")

    print("*********************************************************************************")
    print("************************* Estimate the power consumption ************************")
    print("*********************************************************************************")
