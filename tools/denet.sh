#!/bin/bash
root_path=$(pwd)

function run_onnx_convert() {
    set -v
    modelDir="./runs/train/exp21/weights"
    imageDir="./runs/train/exp21/det_img"
    outDir="${root_path}/${modelDir}/out"
    modelName=detnet
    outNetName=detnet

    inputLayerName="i:images=${outDir}/dra_image_bin/dra_bin_list.txt|is:1,3,352,512|idf:0,0,0,0|iq|im:0,0,0|ic:255.0"
#    outputLayerName0="o:804|odf:fp32"
#    outputLayerName1="o:863|odf:fp32"
#    outputLayerName2="o:922|odf:fp32"
    outputLayerName0="o:326|odf:fp32"
    outputLayerName1="o:385|odf:fp32"
    outputLayerName2="o:444|odf:fp32"

    rm -rf $outDir
    mkdir -m 755 $outDir
    rm -rf $outDir/dra_image_bin
    mkdir -m 755 -p $outDir/dra_image_bin

    #amba
    # source /usr/local/amba-cv-tools-2.2.1-20200928.ubuntu-18.04/env/cv25.env
    source /usr/local/amba-cv-tools-2.4.1.4.1066.ubuntu-18.04/env/cv25.env

    #cuda10
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

    #caffe
    export LD_LIBRARY_PATH=/opt/caffe/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/opt/caffe/python:$PYTHONPATH

    graph_surgery.py onnx -m $modelDir/${modelName}.onnx  -t Default
    mv $modelDir/${modelName}.onnx $modelDir/${modelName}_raw.onnx
    mv $modelDir/${modelName}_modified.onnx $modelDir/${modelName}.onnx

#    ls $imageDir/*.* > $imageDir/img_list.txt
#
#    imgtobin.py -i $imageDir/img_list.txt \
#                -o $outDir/dra_image_bin \
#                -c $inputColorFormat \
#                -d 0,0,0,0 \
#                -s $outputShape
#
#    ls $outDir/dra_image_bin/*.bin > $outDir/dra_image_bin/dra_bin_list.txt

    gen_image_list.py -f $imageDir \
                      -o $imageDir/img_list.txt \
                      -ns -e jpg -c 0 -d 0,0 -r 352,512 \
                      -bf $outDir/dra_image_bin \
                      -bo $outDir/dra_image_bin/dra_bin_list.txt

    rm -rf ${outDir}/out_parser
    onnxparser.py -m $modelDir/${modelName}.onnx \
                  -o $outNetName \
                  -of ${outDir}/out_parser \
                  -isrc ${inputLayerName} \
                  -odst $outputLayerName0 \
                  -odst $outputLayerName1 \
                  -odst $outputLayerName2 \
                  # -c act-allow-fp16,coeff-force-fx16

    # Vas Compiler
    rm -rf ${outDir}/out_parser/vas_output
    cd ${outDir}/out_parser
    vas -auto -show-progress ${outNetName}.vas
    cd -

#    # Run Ades
#    rm -rf ${outDir}/ades_output
#    mkdir -m 755 -p ${outDir}/ades_output
#    ades_autogen.py -v ${modelName}  \
#                    -p ${outDir}/out_parser \
#                    -l ${outDir}/ades_output  \
#                    -ib text_input=$(cat ${outDir}/dra_image_bin/dra_bin_list.txt | head -1)
#    cd ${outDir}/ades_output
#    ades ${modelName}_ades.cmd
#    cd -
#
#    # Run layer_compare.py
#    rm -rf ${outDir}/layer_compare
#    mkdir -m 755 -p ${outDir}/layer_compare
#    layer_compare.py onnx -m ${modelDir}/${modelName}.onnx \
#                          -isrc ${inputLayerName} \
#                          -c act-force-fx16,coeff-force-fx16 \
#                          -odst ${outputLayerName} \
#                          -n ${modelName} \
#                          -v ${outDir}/out_parser \
#                          -o ${outDir}/layer_compare/layer_compare \
#                          -d 3
#    mv lc_cnn_output preproc -t ${outDir}/layer_compare/

    rm -rf ${outDir}/cavalry
    mkdir -m 755 -p ${outDir}/cavalry
    cavalry_gen -d $outDir/out_parser/vas_output/ \
                -f $outDir/cavalry/$outNetName.bin \
                -p $outDir/ \
                -V 2.2.8.2 \
                -v > $outDir/cavalry/cavalry_info.txt
    echo  $(cat ${outDir}/dra_image_bin/dra_bin_list.txt | head -1) | xargs -n 1 echo | xargs -i cp -rf {} ${outDir}/cavalry/

    cp $outDir/cavalry/$outNetName.bin  ${root_path}/${outNetName}.bin
    rm -rf logs lc_cnn_output lc_onnx_output ades
}

function main() {
    if [ -n "$1" ]; then
        dataset_train_path=$1
    else
        dataset_train_path=/easy_ai/ImageSets/train.txt
    fi

    if [ -n "$2" ]; then
        dataset_val_path=$2
    else
        dataset_val_path=/easy_ai/ImageSets/val.txt
    fi
    echo ${dataset_train_path}
    echo ${dataset_val_path}

    rm -rf ./.easy_log/detect2d*

#    CUDA_VISIBLE_DEVICES=0 python3 -m easy_tools.easy_ai --task DeNet --gpu 0 --trainPath ${dataset_train_path} --valPath ${dataset_val_path}
#    if [ $? -ne 0 ]; then
#          echo "Failed to start easy_ai"
#          exit -1
#    fi
#    python3 -m easy_tools.easy_convert --task DeNet --input ./.easy_log/snapshot/denet.onnx
#    if [ $? -ne 0 ]; then
#          echo "Failed to start easy_convert"
#          exit -2
#    fi
    # run_caffe_convert
    # python3 export.py --weights ./runs/train/exp18/weights/best.pt --imgsz 416 --opset 11 --include onnx --simplify
    # graph_surgery.py onnx -m ./runs/train/exp18/weights/best.onnx -o ./runs/train/exp18/weights/best_raw.onnx -on 444,385,326 -t ConstantifyShapes,FoldConstants,CutGraph
    run_onnx_convert
}

main "$1" "$2"