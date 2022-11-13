class QuantOperatorBase:
    '''
    Base class of operator node quantization.
    '''
    def __init__(self, onnx_quantizer, onnx_node):
        '''
        Store the quantizer and target operator node.

        :param onnx_quantizer: Quantizer which contains whole model or subgraph
        :param onnx_node: Target ONNX operator node
        '''
        self.quantizer = onnx_quantizer
        self.node = onnx_node

    def quantize(self):
        '''
        Convert the operator node into quantization-compatible nodes and store them.

        Given a node which does not support quantization, this method checks whether the input to
        this node is quantized and adds a DequantizeLinear node to dequantize this input back to FP32
            parameter node: Current node
        '''

        for _, node_input in enumerate(self.node.input):
            dequantize_node = self.quantizer._dequantize_value(node_input)
            if dequantize_node is not None:
                self.quantizer.new_nodes.append(dequantize_node)
            else:
                pass

        # Append the fp32 operator node
        self.quantizer.new_nodes.append(self.node)