class layer_information():
    def __init__(self):
        """The Layer Information class - handles the layer information """
        self.name           = None
        self.operation_type = None
        self.in_size        = None
        self.out_channels   = None
        self.kernel_size    = None
        self.stride         = None
        self.padding        = None
        self.dilation       = None

    def set_parameters(self, name, op_type, in_size, out_channels, kernel_size, stride, padding, dilation):
        self.name           = name
        self.operation_type = op_type
        self.in_size        = in_size
        self.out_channels   = out_channels
        self.kernel_size    = kernel_size
        self.stride         = stride
        self.padding        = padding
        self.dilation       = dilation

    def __repr__(self):
        return (f'Name: {self.name}, Opertion type: {self.operation_type}, Input size: {self.in_size}, Output channels: {self.out_channels}, Kernel size: {self.kernel_size}, Stride: {self.stride}, Padding: {self.padding}, Dilation: {self.dilation}')

class shape_inferred():
    def __init__(self):
        """The shape inferred class - handles the shape inferred information """
        self.name       = None
        self.dimensions = None

    def set_parameters(self, name, dimensions):
        self.name = name
        self.dimensions = dimensions

    def __repr__(self):
        return (f'Name: {self.name}, Dimensions: {self.dimensions}')
    
class filters_struct():
    def __init__(self):
        """The filters structure class - helps in finding the number of filters for a convolutional layer """
        self.name  = None
        self.value = None

    def set_parameters(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return (f'Name: {self.name}, Dimensions: {self.value}')
    
class User_Input():
    def __init__(self):
        """The User Input class - handles the hardware platform configuration """
        
    def set_parameters(self, pipeline_en, pipeline_stages, lu_en, lu_factor, data_forwarding, memory_latency, addition, multiplication, division, nr_adders, nr_multipliers, SIMD_status, VLIW_status):
        """set_parameters - Configure the main important characteristics of the hardware plaftorm

        Args:
            pipeline_en (str): Pipeline is ENABLED or DISABLED
            pipeline_stages (int): The number of the pipeline stages (e.g. 5, 7)
            data_forwarding (str): Data forwarding is ENABLED or DISABLED
            addition(int): Number of clock cycle to execute one addition instruction
            multiplication(int): Number of clock cycle to execute one multiplication instruction
            division(int): Number of clock cycle to execute one division instruction
            nr_adders (int): Number of functional units that can execute an ADD operation
            nr_multipliers (int): Number of functional units that can execute a MUL operation
            SIMD_status (str): Single Instruction Multiple Data hardware accelerator (ENABLED or DISABLED)

        Returns:
            None:
        """
        self.pipeline_en     = pipeline_en
        self.pipeline_stages = pipeline_stages
        self.data_forwarding = data_forwarding
        self.memory_latency  = memory_latency
        self.addition        = addition
        self.multiplication  = multiplication
        self.division        = division
        self.nr_adders       = nr_adders
        self.nr_multipliers  = nr_multipliers
        self.SIMD_status     = SIMD_status

    def __repr__(self):
        return (f'Pipeline: {self.pipeline_en}, Pipeline stages: {self.pipeline_stages}, Data forwarding: {self.data_forwarding}, Memory latency: {self.memory_latency}, Addition: {self.addition} cc, Multiplication: {self.multiplication} cc, Division: {self.division} cc, Nr. adders: {self.nr_adders}, Nr. multipliers: {self.nr_multipliers}, SIMD: {self.SIMD_status}')
