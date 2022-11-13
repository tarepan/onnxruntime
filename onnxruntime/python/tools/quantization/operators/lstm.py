import onnx
import numpy
from .base_operator import QuantOperatorBase
from ..quant_utils import attribute_to_kwarg, ms_domain, QuantType
from onnx import onnx_pb as onnx_proto
'''
    Quantize LSTM
'''


class LSTMQuant(QuantOperatorBase):
    '''
    Quantize ai.onnx::LSTM standard operator into com.microsoft::DynamicQuantizeLSTM custom operator.
    '''
    def __init__(self, onnx_quantizer, onnx_node):
        '''
        :param onnx_quantizer: Quantizer
        :param onnx_node: ONNX LSTM node
        '''
        super().__init__(onnx_quantizer, onnx_node)

    def quantize(self):
        node = self.node
        assert (node.op_type == "LSTM")

        name_W = node.input[1]
        name_R = node.input[2]
        if (not self.quantizer.is_valid_quantize_weight(name_W)
                or not self.quantizer.is_valid_quantize_weight(name_R)):
            super().quantize()
            return

        model = self.quantizer.model
        W = model.get_initializer(name_W)
        R = model.get_initializer(name_R)

        if (len(W.dims) != 3 or len(R.dims) != 3):
            super().quantize()
            return
        [W_num_dir, W_4_hidden_size, W_input_size] = W.dims
        [R_num_dir, R_4_hidden_size, R_hidden_size] = R.dims

        if self.quantizer.is_per_channel():
            del W.dims[0]
            del R.dims[0]
            W.dims[0] = W_num_dir * W_4_hidden_size
            R.dims[0] = R_num_dir * R_4_hidden_size

        # Static weight quantization
        name_W_q, name_W_q_zp, name_W_q_scale = self.quantizer.quantize_weight_per_channel(
            name_W, onnx_proto.TensorProto.INT8, 0)
        name_R_q, name_R_q_zp, name_R_q_scale = self.quantizer.quantize_weight_per_channel(
            name_R, onnx_proto.TensorProto.INT8, 0)
        W_quant_weight = model.get_initializer(name_W_q)
        R_quant_weight = model.get_initializer(name_R_q)

        # Weight axis swap for DynamicQuantizeLSTM
        W_quant_tranposed = onnx.numpy_helper.from_array(
            onnx.numpy_helper.to_array(W_quant_weight)
                .reshape((W_num_dir, W_4_hidden_size, W_input_size))
                .transpose((0, 2, 1)), # [dir, h, i] => [dir, i, h]
            name_W_q)
        R_quant_tranposed = onnx.numpy_helper.from_array(
            onnx.numpy_helper.to_array(R_quant_weight)
                .reshape((R_num_dir, R_4_hidden_size, R_hidden_size))
                .transpose((0, 2, 1)), # [dir, h, i] => [dir, i, h]
            name_R_q)
        model.remove_initializers([W_quant_weight, R_quant_weight])
        model.add_initializer(W_quant_tranposed)
        model.add_initializer(R_quant_tranposed)

        # Update dimensions of zero-point and scale
        W_quant_zp = model.get_initializer(name_W_q_zp)
        R_quant_zp = model.get_initializer(name_R_q_zp)
        W_quant_scale = model.get_initializer(name_W_q_scale)
        R_quant_scale = model.get_initializer(name_R_q_scale)
        if self.quantizer.is_per_channel():
            W_quant_zp.dims[:] = [W_num_dir, W_4_hidden_size]
            R_quant_zp.dims[:] = [R_num_dir, R_4_hidden_size]
            W_quant_scale.dims[:] = [W_num_dir, W_4_hidden_size]
            R_quant_scale.dims[:] = [R_num_dir, R_4_hidden_size]

        # Update inputs
        inputs = []
        input_len = len(node.input)
        inputs.extend([node.input[0]])
        inputs.extend([quant_input_weight_tuple[0], quant_recurrent_weight_tuple[0]]) # W, R => W_q, R_q
        ## `""` means empty input for optional argument
        inputs.extend([node.input[3] if input_len > 3 else ""])
        inputs.extend([node.input[4] if input_len > 4 else ""])
        inputs.extend([node.input[5] if input_len > 5 else ""])
        inputs.extend([node.input[6] if input_len > 6 else ""])
        inputs.extend([node.input[7] if input_len > 7 else ""])
        inputs.extend([name_W_q_scale, name_W_q_zp, name_R_q_scale, name_R_q_zp]) # additional inputs

        # Transfer and modify node properties
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        quant_lstm_name = "" if node.name == "" else node.name + "_quant"

        # Add a DynamicQuantizeLSTM node as quantized LSTM
        quant_lstm_node = onnx.helper.make_node("DynamicQuantizeLSTM", inputs, node.output, quant_lstm_name, **kwargs)
        self.quantizer.new_nodes.append(quant_lstm_node)

        # Input DQ if needed
        dequantize_node = self.quantizer._dequantize_value(node.input[0])
        if dequantize_node is not None:
            self.quantizer.new_nodes.append(dequantize_node)
