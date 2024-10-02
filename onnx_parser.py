# Some standard imports
import numpy as np
import warnings
import onnx
from onnx import shape_inference
from google.protobuf.json_format import MessageToDict
from utils import *

class ParserWarning(UserWarning):
    pass

def extract_input_dimensions(model):
    """
    Extract the input size of the NN model.

    Args:
        model (struct): The neural network model expressed in ONNX format. 

    Returns:
        int: Input size of the NN model (input for the first layer).
    """
    input_size_list = []

    # Extract the input dimensions
    for _input in model.graph.input:
        input_dim = _input.type.tensor_type.shape.dim
        for dimension in input_dim:
            input_size_list.append(dimension.dim_value)

        nn_input_size = np.flip(np.array(input_size_list))
        input_size_list.clear()

    return nn_input_size

def compute_shape_inference(model):
    """
    Compute the shape inferrence for the given modelto obtain the input
    size for each layer in the network.

    Args:
        model (struct): The neural network model expressed in ONNX format. 

    Returns:
        (struct): The inferred neural network model. 
        
    """
    
    inferred_model = shape_inference.infer_shapes(model)

    return inferred_model

# Helper function to extract the dimensions after the shape inferred process
def extract_shape_inferred_dimensions(model):
    # List with all inferred shapes
    shape_list      = []
    # List with all input sizes
    input_size_list = []
    
    for _layer in model.graph.value_info:
        # Create a shape_inferred object to store the informations
        shape_info = shape_inferred()
        # Extract the name of the layer
        shape_info.name = '[\'' + str(_layer.name) + '\']'
        # Extract the input dimension of the next layer
        shape = _layer.type.tensor_type.shape.dim
        for dimension in shape:
            input_size_list.append(dimension.dim_value)
        shape_info.dimensions = np.flip(np.array(input_size_list))
        input_size_list.clear()
        shape_list.append(shape_info)
    
    return shape_list

def extract_conv_filters(model):
    """
    Helper function to extract the number of filters for a convolutional layer

    Args:
        model (struct): The neural network model expressed in ONNX format.  

    Returns:
        (list): The number of filters for each convolutional layer. 
        
    """

    # List with all convolutional filters
    filters_list = []

    # Iterate and extract the layers parameters:
    for _input in model.graph.initializer:
        if "conv" in str({_input.name}) and  "bias" in str({_input.name}) :
            text_name = _input.name
            element = filters_struct()
            element.set_parameters(text_name.split(".", 1)[0], _input.dims)
            filters_list.append(element)
    return filters_list

def parse_nn_model(model):
    """
    Main method to parse the neural network model expressed in ONNX format

    Args:
        model (struct): The neural network model expressed in ONNX format.  

    Returns:
        (list): The layers of the model (with the correct input size). 
        
    """
    # List with all layers information
    layers = []

    # Iterate and extract the layers parameters:
    for _input in model.graph.node:
        # Create a new layer object
        layer_info = layer_information()

        # Convolutional layer
        if _input.op_type == "Conv":
            layer_info.operation_type = "Conv"
            layer_info.name = str(_input.output)

            # Extract the conv2D parameters
            attr_list = _input.attribute
            for attribute in attr_list:
                if attribute.name == "dilations":
                    layer_info.dilation = attribute.ints
                elif attribute.name == "kernel_shape":
                    layer_info.kernel_size = attribute.ints
                elif attribute.name == "pads":
                    layer_info.padding = attribute.ints
                elif attribute.name == "strides":
                    layer_info.stride = attribute.ints
                
            # Put the element in the list
            layers.append(layer_info)

        # MaxPool layer
        elif  _input.op_type == "MaxPool":
            layer_info.operation_type = "MaxPool"
            layer_info.name = str(_input.output)

            # Extract the MaxPool2D parameters
            attr_list = _input.attribute
            for attribute in attr_list:
                if attribute.name == "kernel_shape":
                    layer_info.kernel_size = attribute.ints
                elif attribute.name == "pads":
                    layer_info.padding = attribute.ints
                elif attribute.name == "strides":
                    layer_info.stride = attribute.ints

            # Put the element in the list
            layers.append(layer_info)

        # AveragePool layer
        elif  _input.op_type == "AveragePool":
            layer_info.operation_type = "AvgPool"
            layer_info.name = str(_input.output)

            # Extract the AvgPool2D parameters
            attr_list = _input.attribute
            for attribute in attr_list:
                if attribute.name == "kernel_shape":
                    layer_info.kernel_size = attribute.ints
                elif attribute.name == "pads":
                    layer_info.padding = attribute.ints
                elif attribute.name == "strides":
                    layer_info.stride = attribute.ints

            # Put the element in the list
            layers.append(layer_info)

        # Linear - fully connected layer
        elif  _input.op_type == "Gemm":
            layer_info.operation_type = "Gemm"
            layer_info.name = str(_input.output)

            # Put the element in the list
            layers.append(layer_info)

        # ReLU - activation function
        elif  _input.op_type == "Relu":
            layer_info.operation_type = "ReLU"
            layer_info.name = str(_input.output)

            # Put the element in the list
            layers.append(layer_info)

        # Not supported layers
        else:
            warnings.warn(f"\n Warning <-> Layer {_input.op_type} is not supported", ParserWarning, stacklevel = 2)

    return layers

def append_dimensions(input_arr, shape_list, layers):
    """
    Helper function to append the correct input size for each layer

    Args:
        input_arr (int): The input of the neural network
        shape_list (list): The number of input size for each layer. 
        layers (list): The layers of the model without the number of filters. 

    Returns:
        (list): The updated layers of the model (with the correct input size). 
        
    """
    # Take the dimension of the first layer from the input
    layers[0].in_size = input_arr

    for index in range(1, len(layers)):
        # Iterate through the shape_list variable and extract the correct dimensions
        for index2 in range(len(shape_list)):
            if layers[index].name == shape_list[index2].name:
                # The input dimension of the current layer is equal to the output dimension of the previous layer
                layers[index].in_size = shape_list[index2 - 1].dimensions

    return layers

def append_filters(filter_list, layers):
    """
    Helper function to append the information regarding the filters in convolution layer

    Args:
        filter_list (list): The number of output channels for each convolution layer. 
        layers (list): The layers of the model without the number of filters. 

    Returns:
        (list): The updated layers of the model (with the number of convolutional filters). 
        
    """
    for index in range(len(layers)):
        # Iterate through the filter_list variable and extract the correct number of filters
        for index2 in range(len(filter_list)):
            if layers[index].name.split("/", 2)[1] == filter_list[index2].name:
                # The out_channels for the current CONV layer is equal to the value of the filter_list element
                layers[index].out_channels = filter_list[index2].value

    return layers


def load_and_parse(name):
    """
    Main function to be used for loading and parsing a neural network model.

    Args:
        name (str): The name of the neural network. 

    Returns:
        (list): The layers of the model with all required information extracted. 
        
    """
    # Load the NN model
    model = onnx.load("nn_models/" + name)

   # Run the shape inference to get the node dimensions (e.g. channels)
    inferred_model = compute_shape_inference(model)

    # Extract the input size of the NN model
    nn_input_size = extract_input_dimensions(model)

    # Extract the shape_inferred dimensions
    shape_list = extract_shape_inferred_dimensions(inferred_model)

    # Extract the convolutional filters
    conv_filter_list = extract_conv_filters(model)

    # Extract the operation type and other parameters
    layers = parse_nn_model(model)

    # Append the dimension information for each layer in the network
    layers = append_dimensions(nn_input_size, shape_list, layers)

    # Append the number of filters for each convolutional layer in the network
    layers = append_filters(conv_filter_list, layers)

    return layers