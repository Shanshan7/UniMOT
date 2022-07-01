import onnx
onnx.utils.extract_model('fairmot_1088x1088.onnx', 'UniMOT_1088x1088.onnx', ['im_shape', 'image', 'scale_factor'], ['concat_8.tmp_0', 'gather_5.tmp_0', 'tmp_26', 'tmp_27', 'tmp_28', 'tmp_29'])
