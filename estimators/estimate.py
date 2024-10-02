# Function to estimate the number of clock cycles for a convolutional layer
def calculate_conv_cc(layer_info, processor_info):
    """
    Calculates the number of clock cycles for a convolution layer.

    Args:
        layer_info (struct): Contains all relevant information for a specific layer in the architecture. 
                             See more details in utils.py file
        processor_info (struct): Describe the processor configuration. 
                             See more details in utils.py file
        cc_per_instruction (int): If pipeline is not present in the system, then each instruction takes X clock cycle to execute

    Returns:
        int: Total number of clock cycles.
    """

    # Extract the information from the layer structure
    input_size     = layer_info.in_size
    kernel_size    = layer_info.kernel_size
    stride         = layer_info.stride
    padding        = layer_info.padding
    dilation       = layer_info.dilation
    output_filters = layer_info.out_channels

    # Extract the information from the processor structure
    pipeline_en     = processor_info.pipeline_en
    data_forwarding = processor_info.data_forwarding
    multiply_cc     = processor_info.multiplication
    add_cc          = processor_info.addition
    load_op_cc      = processor_info.memory_latency
    store_op_cc     = processor_info.memory_latency

    # Calculate the output size
    h_out = (input_size[0] - kernel_size[0] + 2 * padding[0]) / stride[0] + 1
    w_out = (input_size[1] - kernel_size[1] + 2 * padding[0]) / stride[0] + 1
    total_positions = h_out * w_out

    # Calculate the number of instructions per kernel
    mul_operation_per_kernel = kernel_size[0] * kernel_size[1]

    # Don't forget the add instructions
    add_operation_per_kernel = mul_operation_per_kernel - 1

    # Calculate the total number of operations per filter
    mul_per_filter = total_positions * mul_operation_per_kernel
    add_per_filter = total_positions * add_operation_per_kernel

    # Calculate the total number of instructions
    total_mul = mul_per_filter * output_filters * input_size[2]
    total_add = add_per_filter * output_filters * input_size[2]
    total_ops = total_mul + total_add

    if pipeline_en == "Enabled" and data_forwarding == "Enabled":
        # Number of CC to load the kernel_matrix
        total_load_kernel_cc = kernel_size[0] * kernel_size[1] * load_op_cc

        # Add the remaining CC for calculating the conv_product for one element
        total_one_element_cc  = (kernel_size[0] * kernel_size[1]) * (load_op_cc + multiply_cc)
        total_one_element_cc += (kernel_size[0] * kernel_size[1] - 1) * add_cc

        # Number of CC to store the result
        total_one_element_cc += store_op_cc

        # Calculate the CC for the remaining positions
        total_elements_cc = (total_positions - 1) * (total_one_element_cc - load_op_cc)
        total_cc = total_one_element_cc +  total_elements_cc

        # Multiply with the number of filters
        total_cc = (total_cc + total_load_kernel_cc) * output_filters

        # Multiply with the number of channels
        total_cc = total_cc * input_size[2]

    return total_cc

# Function to estimate the number of clock cycles for ReLU
def calculate_relu_cc(layer_info, processor_info, cc_per_instruction):
    """
    Calculates the number of clock cycles for ReLU activation function.

    Args:
        layer_info (struct): Contains all relevant information for a specific layer in the architecture. 
                             See more details in utils.py file
        processor_info (struct): Describe the processor configuration. 
                             See more details in utils.py file
        cc_per_instruction (int): If pipeline is not present in the system, then each instruction takes X clock cycle to execute

    Returns:
        int: Total number of clock cycles.
    """
    # Extract the information from the layer structure
    input_size     = layer_info.in_size

    # Extract the information from the processor structure
    pipeline_en     = processor_info.pipeline_en
    load_op_cc      = processor_info.memory_latency
    store_op_cc     = processor_info.memory_latency

    total_positions = input_size[0] * input_size[1] * input_size[2]

    if pipeline_en == "Enabled":
        total_cc = total_positions * (load_op_cc + 2 + store_op_cc)

    elif pipeline_en == "Disabled":
        # Number of CC for CMP and JMP instructions
        total_cc = total_positions * 2 * cc_per_instruction

        # Add the CC for the LOAD instructions 
        total_cc += total_positions * load_op_cc

        # Add the CC for the LOAD instructions 
        total_cc += total_positions * store_op_cc
    return total_cc


# Function to estimate the number of clock cycles for a MaxPool2D layer
def calculate_max_pool_cc(layer_info, processor_info, cc_per_instruction):
    """
    Calculates the number of clock cycles for a max pooling layer.

    Args:
        layer_info (struct): Contains all relevant information for a specific layer in the architecture. 
                             See more details in utils.py file
        processor_info (struct): Describe the processor configuration. 
                             See more details in utils.py file
        cc_per_instruction (int): If pipeline is not present in the system, then each instruction takes X clock cycle to execute

    Returns:
        int: Total number of clock cycles.
    """

    # Extract the information from the layer structure
    input_size     = layer_info.in_size
    kernel_size    = layer_info.kernel_size
    stride         = layer_info.stride
    padding        = layer_info.padding

    # Extract the information from the processor structure
    pipeline_en     = processor_info.pipeline_en
    data_forwarding = processor_info.data_forwarding
    load_op_cc      = processor_info.memory_latency
    store_op_cc     = processor_info.memory_latency

    # Calculate the output size
    h_out = (input_size[0] - kernel_size[0] + 2 * padding[0]) / stride[0] + 1
    w_out = (input_size[1] - kernel_size[1] + 2 * padding[0]) / stride[0] + 1
    total_positions = h_out * w_out

    if pipeline_en == "Enabled" and data_forwarding == "Enabled":
        # total_cc = 2 x MOV + CMP + JGE + MOV 
        total_cc = 2 * load_op_cc  + 1 + 1 + 1

        # Add the remaining CC for calculating the conv_product for one element
        total_cc += (kernel_size[0] * kernel_size[1] - 2) * (load_op_cc + 1 + 1 + 1)

        # Number of CC to store the result
        total_cc += store_op_cc

        # Multiply with the number of elements
        total_cc = total_cc * total_positions

        # Multiply with the number of channels
        total_cc = total_cc * input_size[2]

    elif pipeline_en == "Disabled":
        # total_cc = total_positions * ((kernel_size *kernel_size - 2) * (MOV + CMP + JGE + MOV) + MOV + MOV + CMP + JGE + MOV
        total_cc = total_positions * 4 * cc_per_instruction

    return total_cc

# Function to estimate the number of clock cycles for a AvgPool2D layer
def calculate_avg_pool_cc(layer_info, processor_info, cc_per_instruction):
    """
    Calculates the number of clock cycles for an average pooling layer.

    Args:
        layer_info (struct): Contains all relevant information for a specific layer in the architecture. 
                             See more details in utils.py file
        processor_info (struct): Describe the processor configuration. 
                             See more details in utils.py file
        cc_per_instruction (int): If pipeline is not present in the system, then each instruction takes X clock cycle to execute

    Returns:
        int: Total number of clock cycles.
    """

    # Extract the information from the layer structure
    input_size     = layer_info.in_size
    kernel_size    = layer_info.kernel_size
    stride         = layer_info.stride
    padding        = layer_info.padding

    # Extract the information from the processor structure
    pipeline_en     = processor_info.pipeline_en
    data_forwarding = processor_info.data_forwarding
    add_cc          = processor_info.addition
    div_cc          = processor_info.division
    load_op_cc      = processor_info.memory_latency
    store_op_cc     = processor_info.memory_latency

    # Calculate the output size
    h_out = (input_size[0] - kernel_size[0] + 2 * padding[0]) / stride[0] + 1
    w_out = (input_size[1] - kernel_size[1] + 2 * padding[0]) / stride[0] + 1
    total_positions = h_out * w_out

    if pipeline_en == "Enabled" and data_forwarding == "Enabled":
        # total_cc = MOV + INC
        total_cc = load_op_cc + 1

        # Add the remaining CC for calculating the conv_product for one element
        total_cc += (kernel_size[0] * kernel_size[1] - 1) * (load_op_cc + add_cc + 1) + div_cc

        # Number of CC to store the result
        total_cc += store_op_cc

        # Multiply with the number of elements
        total_cc = total_cc * total_positions

        # Multiply with the number of channels
        total_cc = total_cc * input_size[2]

    elif pipeline_en == "Disabled":
        # Number of CC for the LOAD instructions
        total_cc = total_positions * kernel_size[0] * kernel_size[1] * load_op_cc
        # Number of CC for the STORE instructions
        total_cc += total_positions * store_op_cc
        # Number of CC for the remaining instructions (INC, ADD, DIV)
        total_cc += total_positions * (2 * kernel_size[0] * kernel_size[1]) * cc_per_instruction

    return total_cc

# Function to estimate the number of clock cycles for a Dense layer
def calculate_fully_connected_cc(layer_info, processor_info, cc_per_instruction):
    """
    Calculates the number of clock cycles for a fully connected layer.

    Args:
        layer_info (struct): Contains all relevant information for a specific layer in the architecture. 
                             See more details in utils.py file
        processor_info (struct): Describe the processor configuration. 
                             See more details in utils.py file
        cc_per_instruction (int): If pipeline is not present in the system, then each instruction takes X clock cycle to execute

    Returns:
        int: Total number of clock cycles.
    """

    # Extract the information from the layer structure
    in_features     = layer_info.in_size
    out_features    = layer_info.out_size

    # Extract the information from the processor structure
    pipeline_en     = processor_info.pipeline_en
    data_forwarding = processor_info.data_forwarding
    add_cc          = processor_info.addition
    multiply_cc     = processor_info.multiplication
    load_op_cc      = processor_info.memory_latency
    store_op_cc     = processor_info.memory_latency

    if pipeline_en == "Enabled" and data_forwarding == "Enabled":
        # Load the bias term
        total_cc = load_op_cc

        # Number of CC to calculate the first product
        total_cc += 2 * load_op_cc + multiply_cc + add_cc

        # Add the remaining CC for calculating the results for one output sample
        total_cc += (in_features - 1) * (2 * load_op_cc + multiply_cc + add_cc)

        # Number of CC to store the result
        total_cc += store_op_cc

        # Multiply with the number of elements
        total_cc = total_cc * out_features
        
    elif pipeline_en == "Disabled":
        total_mul = in_features * out_features

        # The number od ADD instruction is equal to the MUL instr (due to the bias term)
        total_add = total_mul
        total_ops = total_mul + total_add

        # Calculate the total number of clock cycle
        total_cc = 2 * load_op_cc * total_mul + out_features * (load_op_cc + store_op_cc) + total_ops * cc_per_instruction

    return total_cc

def calculate_batch_normalization_cc(layer_info, processor_info, cc_per_instruction):
    """
    Calculates the number of clock cycles for a batch normalizaiton layer.

    Args:
        layer_info (struct): Contains all relevant information for a specific layer in the architecture. 
                             See more details in utils.py file
        processor_info (struct): Describe the processor configuration. 
                             See more details in utils.py file
        cc_per_instruction (int): If pipeline is not present in the system, then each instruction takes X clock cycle to execute

    Returns:
        int: Total number of clock cycles.
    """

    # Extract the information from the layer structure
    input_size     = layer_info.in_size

    # Extract the information from the processor structure
    pipeline_en     = processor_info.pipeline_en
    data_forwarding = processor_info.data_forwarding
    add_cc          = processor_info.addition
    multiply_cc     = processor_info.multiplication
    div_cc          = processor_info.division
    load_op_cc      = processor_info.memory_latency
    store_op_cc     = processor_info.memory_latency
    
    sqrt_cc = 1

    # Call the calculate_average_cc 
    total_cc_aux = calculate_average_cc(layer_info, processor_info, cc_per_instruction)

    # Call the calculate_variance_cc
    total_cc_aux += calculate_variance_cc(layer_info, processor_info, cc_per_instruction)

    if pipeline_en == "Enabled" and data_forwarding == "Enabled":
        # Calculate the divisor
        total_cc = 2 * load_op_cc + add_cc + sqrt_cc

        # Number of CC to calculate one element (SUB and MOV are 1cc)
        total_cc += 4 * load_op_cc + 1 + div_cc + 1 + multiply_cc + add_cc

        # Number of CC to store the result
        total_cc += store_op_cc

        #  Multiply with the number of elements
        total_cc = input_size[0] * input_size[1] * total_cc

        # Add the auxiliary cycles for calculating the mean and standard deviation
        total_cc += total_cc_aux
        
    elif pipeline_en == "Disabled":
        print("Not implemented yet!")
    return total_cc

# Helper functions to calculate the number of clock cycles required for finding the average
def calculate_average_cc(layer_info, processor_info, cc_per_instruction):
    """
    Calculates the number of clock cycles for finding the average of an input tensor.

    Args:
        layer_info (struct): Contains all relevant information for a specific layer in the architecture. 
                             See more details in utils.py file
        processor_info (struct): Describe the processor configuration. 
                             See more details in utils.py file
        cc_per_instruction (int): If pipeline is not present in the system, then each instruction takes X clock cycle to execute

    Returns:
        int: Total number of clock cycles.
    """

    # Extract the information from the layer structure
    input_size     = layer_info.in_size

    # Extract the information from the processor structure
    pipeline_en     = processor_info.pipeline_en
    data_forwarding = processor_info.data_forwarding
    add_cc          = processor_info.addition
    div_cc          = processor_info.division
    load_op_cc      = processor_info.memory_latency
    store_op_cc     = processor_info.memory_latency

    total_positions = input_size[0] * input_size[1]

    if pipeline_en == "Enabled" and data_forwarding == "Enabled":
        # Copy the content of the first element
        total_cc = load_op_cc + 1 + (total_positions - 1) * (load_op_cc + add_cc + 1) + div_cc + store_op_cc

    elif pipeline_en == "Disabled":
        # Number of CC for ADD, INC, MOV instructions
        total_cc = (total_positions * 2 - 1) * cc_per_instruction

        # Number of CC for exponential procedure instructions
        total_cc += div_cc

        # Add the CC for the LOAD instructions 
        total_cc += total_positions * load_op_cc

        # Add the CC for the LOAD instructions 
        total_cc += store_op_cc

    return total_cc

# Helper functions to calculate the number of clock cycles required for calculating the variance
def calculate_variance_cc(layer_info, processor_info, cc_per_instruction):
    """
    Calculates the number of clock cycles for calculating the variance of an input tensor.

    Args:
        layer_info (struct): Contains all relevant information for a specific layer in the architecture. 
                             See more details in utils.py file
        processor_info (struct): Describe the processor configuration. 
                             See more details in utils.py file
        cc_per_instruction (int): If pipeline is not present in the system, then each instruction takes X clock cycle to execute

    Returns:
        int: Total number of clock cycles.
    """

    # Extract the information from the layer structure
    input_size     = layer_info.in_size

    # Extract the information from the processor structure
    pipeline_en     = processor_info.pipeline_en
    data_forwarding = processor_info.data_forwarding
    add_cc          = processor_info.addition
    multiply_cc     = processor_info.multiplication
    div_cc          = processor_info.division
    load_op_cc      = processor_info.memory_latency
    store_op_cc     = processor_info.memory_latency

    total_positions = input_size[0] * input_size[1]

    # Call the calculate_average_cc first
    total_cc = calculate_average_cc(layer_info, processor_info, cc_per_instruction)
    
    if pipeline_en == "Enabled" and data_forwarding == "Enabled":
        # Copy the content of the first element
        total_cc += 2 * load_op_cc + 1 + 1 + multiply_cc + 1 + 1

        #  Remaining elements in the input tensor
        total_cc += (total_positions - 1) * (load_op_cc + 1 + 1 + multiply_cc + add_cc + 1) + div_cc + store_op_cc

    elif pipeline_en == "Disabled":
        # Number of CC for ADD, INC, MOV, SUB instructions
        total_cc += (total_positions * 4) * cc_per_instruction

        # Number of CC for exponential procedure instructions
        total_cc += div_cc

        # Add the CC for the LOAD instructions 
        total_cc += (total_positions + 1) * load_op_cc

        # Add the CC for the LOAD instructions 
        total_cc += store_op_cc

    return total_cc


# Function to estimate the number of clock cycles for SoftMax
def calculate_softmax_cc(layer_info, processor_info, cc_per_instruction):
    """
    Calculates the number of clock cycles for SoftMax activation function.

    Args:
        layer_info (struct): Contains all relevant information for a specific layer in the architecture. 
                             See more details in utils.py file
        processor_info (struct): Describe the processor configuration. 
                             See more details in utils.py file
        cc_per_instruction (int): If pipeline is not present in the system, then each instruction takes X clock cycle to execute

    Returns:
        int: Total number of clock cycles.
    """

    # Extract the information from the layer structure
    input_size     = layer_info.in_size

    # Extract the information from the processor structure
    pipeline_en     = processor_info.pipeline_en
    data_forwarding = processor_info.data_forwarding
    add_cc          = processor_info.addition
    div_cc          = processor_info.division
    load_op_cc      = processor_info.memory_latency
    store_op_cc     = processor_info.memory_latency

    exp_cc = 1

    total_positions = input_size[0] * input_size[1] * input_size[2]

    if pipeline_en == "Enabled" and data_forwarding == "Enabled":
        # Calculate the first exp term and store in another register
        total_cc = load_op_cc + exp_cc + 1 
        # Calculate the sum exp
        total_cc += (total_positions - 1) * (load_op_cc + exp_cc + add_cc) + div_cc + store_op_cc

        # Calculate the remaining elements
        total_cc += (total_positions - 1) * (load_op_cc + exp_cc + div_cc + store_op_cc)

    elif pipeline_en == "Disabled":
        # Number of CC for NEG, INC, MOV instructions
        total_cc = total_positions * 3 * cc_per_instruction

        # Number of CC for exponential procedure instructions
        total_cc += total_positions * exp_cc

        # Number of CC for exponential procedure instructions
        total_cc += total_positions * div_cc

        # Add the CC for the LOAD instructions 
        total_cc += total_positions * load_op_cc

        # Add the CC for the LOAD instructions 
        total_cc += total_positions * store_op_cc
    return total_cc

# Function to estimate the number of clock cycles for Sigmoid
def calculate_sigmoid_cc(layer_info, processor_info, cc_per_instruction):
    """
    Calculates the number of clock cycles for sigmoid activation function.

    Args:
        layer_info (struct): Contains all relevant information for a specific layer in the architecture. 
                             See more details in utils.py file
        processor_info (struct): Describe the processor configuration. 
                             See more details in utils.py file
        cc_per_instruction (int): If pipeline is not present in the system, then each instruction takes X clock cycle to execute

    Returns:
        int: Total number of clock cycles.
    """

    # Extract the information from the layer structure
    input_size     = layer_info.in_size

    # Extract the information from the processor structure
    pipeline_en     = processor_info.pipeline_en
    data_forwarding = processor_info.data_forwarding
    div_cc          = processor_info.division
    load_op_cc      = processor_info.memory_latency
    store_op_cc     = processor_info.memory_latency

    exp_cc = 1

    total_positions = input_size[0] * input_size[1] * input_size[2]

    if pipeline_en == "Enabled" and data_forwarding == "Enabled":
        total_cc = total_positions * (load_op_cc + 1 + exp_cc + 1 + 1 + div_cc + store_op_cc)

    elif pipeline_en == "Disabled":
        # Number of CC for NEG, INC, MOV instructions
        total_cc = total_positions * 3 * cc_per_instruction

        # Number of CC for exponential procedure instructions
        total_cc += total_positions * exp_cc

        # Number of CC for exponential procedure instructions
        total_cc += total_positions * div_cc

        # Add the CC for the LOAD instructions 
        total_cc += total_positions * load_op_cc

        # Add the CC for the LOAD instructions 
        total_cc += total_positions * store_op_cc
    return total_cc

# Function to estimate the number of clock cycles for Dropout
def calculate_dropout_cc(layer_info, processor_info):
    """
    Calculates the number of clock cycles for a dropout layer.

    Args:
        layer_info (struct): Contains all relevant information for a specific layer in the architecture. 
                             See more details in utils.py file
        processor_info (struct): Describe the processor configuration. 
                             See more details in utils.py file

    Returns:
        int: Total number of clock cycles.
    """
    print("Not implemented yet")

# Function to estimate the number of clock cycles for Flatten
def calculate_flatten_cc(layer_info, processor_info):
    """
    Calculates the number of clock cycles for a dropout layer.

    Args:
        layer_info (struct): Contains all relevant information for a specific layer in the architecture. 
                             See more details in utils.py file
        processor_info (struct): Describe the processor configuration. 
                             See more details in utils.py file

    Returns:
        int: Total number of clock cycles.
    """
    print("Not implemented yet")