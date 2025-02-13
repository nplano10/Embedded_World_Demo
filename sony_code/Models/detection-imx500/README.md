# Sony AITRIOS Detection Vision and Sensing Application

## Content

Files
 - `vision_app_objectdetection_v1.1.2.aot`: Object Detection Vision Application, signed and ready for deployment.
 - `vision_app_objectdetection_v1.1.2.wasm`: Object Detection Vision Application, source of the AoT file.
 - `ObjectDetectionPPLParameterSample.json`: PPL Parameter sample for Object Detection.
 - `<BrainBuilder_brain_file>.zip`: Zipped Brain files
    - `<Brain>.keras:` Neurala AI model, quantized, ready for upload on Console
    - `deploy.json`: Neurala's brain configuration metadata
    - `labels.txt`: Class names information
    - `packerOut.zip`, `network.fpk`, `network.pkg`: AI models already converted for Console-incompatible IMX500-equipped devices
 - `README.md`

> NOTE: `<BrainBuilder_brain_file>`, `<Brain>` names will match the names provided in BrainBuilder.


## How to use:

1. Import the `keras` model from this brain (found in the `<BrainBuilder_brain_file>.zip` file)
2. Create a configuration for this model
3. Deploy configuration to your camera.

> Refer to Console user manual, the section "3.5.Create model" and "3.5.2.Import button" on how to import the AI model.
>    - [Console user manual (Japanese)][1]
>    - [Console user manual (English)][2]

4. Upload the WASM application `vision_app_objectdetection_v1.1.2.wasm` to Console
   > Details of this procedure are explained in:
   > - Console user manual for GUI Operation ([jp][1], [en][2])
   > - AITRIOS SDK repository for [scripted Operation][3]
5. Create a CommandParameters configuration.
   - Use the `ObjectDetectionPPLParameterSample.json` to populate `PPLParameter`. See the Neurala knowledge
   base for a [description of the parameters effects][4]. Here is a quick summary:
     * `input_width`, `input_height`: Defines the source image shape. Used to compute bounding boxes coordinates.
     * `dnn_output_detections`: Model maximum number of bounding boxes provided by the brain. For VIA 24.09,
       this value should be 300. For earlier VIA versions, consult the size of the `detection_boxes` output in
       `deploy.json`.  <!-- See VIA-621 -->
     * `max_detections`: Maximum amount of bounding boxes _you_ want to receive in your application.
     * `threshold`: Determine how confident a box needs to be to appear. Higher means fewer boxes.
     > Do not change "header" field
   - Refer to Console User Manual "Appendix A - A.1. Command Parameter file (JSON) specifications"
   (Console Manual [jp][1], [en][2]) for more information.
6. Retrieve and Parse the output data. Use the [SSS AITRIOS Deserialization Sample][5] as reference.

## Reference links

[1]: https://developer.aitrios.sony-semicon.com/documents/console-user-manual
[2]: https://developer.aitrios.sony-semicon.com/en/documents/console-user-manual
[3]: https://github.com/SonySemiconductorSolutions/aitrios-sdk-vision-sensing-app/blob/main/tutorials/4_prepare_application/3_deploy_to_device/README.md
[4]: https://support.neurala.com/docs/brain-builder-for-aitrios#detector
[5]: https://github.com/SonySemiconductorSolutions/aitrios-sdk-deserialization-sample
