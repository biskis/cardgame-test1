import React, {useEffect, useRef, useState} from 'react';
import {StyleSheet, View, Text, Button, Image} from 'react-native';
import {
  Camera,
  useCameraDevice,
  useCameraFormat,
} from 'react-native-vision-camera';
import {useTensorflowModel} from 'react-native-fast-tflite';
// import {useResizePlugin} from 'vision-camera-resize-plugin';
// import { Canvas, Circle, useCanvasRef } from '@shopify/react-native-skia';
import ImageResizer from '@bam.tech/react-native-image-resizer';
import {convertToRGB} from 'react-native-image-to-rgb';

const numChannels = 56;
const imageH = 640;
const imageW = 640;
const numDetections = 8400;

const classLabels = [
  '10C',
  '10D',
  '10H',
  '10S',
  '2C',
  '2D',
  '2H',
  '2S',
  '3C',
  '3D',
  '3H',
  '3S',
  '4C',
  '4D',
  '4H',
  '4S',
  '5C',
  '5D',
  '5H',
  '5S',
  '6C',
  '6D',
  '6H',
  '6S',
  '7C',
  '7D',
  '7H',
  '7S',
  '8C',
  '8D',
  '8H',
  '8S',
  '9C',
  '9D',
  '9H',
  '9S',
  'AC',
  'AD',
  'AH',
  'AS',
  'JC',
  'JD',
  'JH',
  'JS',
  'KC',
  'KD',
  'KH',
  'KS',
  'QC',
  'QD',
  'QH',
  'QS',
];


const CameraDetect = () => {
  const device = useCameraDevice('back');
  const camera = useRef<Camera>(null);

  // const { resize } = useResizePlugin();
  const objectDetection = useTensorflowModel(
    require('../../assets/ai/best_float32.tflite'),
  );
  const model =
    objectDetection.state === 'loaded' ? objectDetection.model : undefined;
  const [image, setImage] = useState(null);
  const [imageConvert, setImageConvert] = useState<string>();
  const [foundObjects, setFoundObjects] = useState([]);

  const handleCapture = async () => {
    if (device) {
      const photo = await camera.current.takePhoto();
      await resizeImage(photo);
    }
  };

  const resizeImage = async (photo: {path: string}) => {
    console.log('Start resize image', photo);
    const resizedImage = await ImageResizer.createResizedImage(
      photo.path,
      imageW,
      imageH,
      'JPEG',
      100,
      90,
      undefined,
      true,
      {mode: 'stretch'},
    );
    console.log('Image resized', resizedImage);
    setImage(resizedImage);
  };

  useEffect(() => {
    console.log('Start processing image', image);
    processImage(image).then();
  }, [image]);

  const convertSnapshotToTensor = async (image: {
    path: string;
    width: number;
    height: number;
  }) => {
    const convertedArray = await convertToRGB(`file://${image.path}`);
    // const convertedArray = await convertToRGB('../../assets/example.jpg');

    // convert to Uint8 array buffer (but some models require float32 format)
    const arrayBuffer = new Float32Array(convertedArray);
    return arrayBuffer;
  };
  const convertSnapshotToTensorOld = async (snapshot: any) => {
    const {width, height, path} = snapshot;
    const imagePath = require('../../assets/example.jpg');
    // const result = await fetch(imagePath);
    const result = await fetch(`file://${path}`);
    const blob = await result.blob();

    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = () => {
        const arrayBuffer = reader.result as ArrayBuffer;
        const uint8Array = new Uint8Array(arrayBuffer); // Get the raw bytes

        const tensor = new Float32Array(width * height * 3);
        let tensorIndex = 0;

        // Assuming RGB format. Each pixel has 3 bytes.
        for (let i = 0; i < uint8Array.length; i += 3) {
          if (tensorIndex >= tensor.length) break; //Prevent out of bounds errors.
          const r = uint8Array[i] / 255;
          const g = uint8Array[i + 1] / 255;
          const b = uint8Array[i + 2] / 255;

          tensor[tensorIndex++] = r;
          tensor[tensorIndex++] = g;
          tensor[tensorIndex++] = b;
        }

        resolve(tensor);
      };

      reader.onerror = error => {
        reject(error);
      };

      reader.readAsArrayBuffer(blob); // Read the blob as an ArrayBuffer
    });
  };

  function parseYOLOOutput(
    outputTensor,
    confidenceThreshold = 0.5,
    iouThreshold = 0.5,
  ) {
    // Assuming outputTensor is a Float32Array

    // Reshape the tensor
    const detections = [];
    for (let i = 0; i < numDetections; i++) {
      const row = [];
      for (let j = 0; j < numChannels; j++) {
        row.push(outputTensor[i * numChannels + j]);
      }
      detections.push(row);
    }

    // Filter detections based on confidence
    const filteredDetections = detections.filter(detection => detection[4] > confidenceThreshold);

    // Non-Maximum Suppression (NMS)
    const nmsDetections = nonMaximumSuppression(filteredDetections, iouThreshold);

    // Parse and format the results
    const results = nmsDetections.map(detection => {
      // Extract bounding box coordinates
      const xCenter = detection[0];
      const yCenter = detection[1];
      const width = detection[2];
      const height = detection[3];

      const xMin = (xCenter - width / 2) * imageW;
      const yMin = (yCenter - height / 2) * imageH;
      const xMax = (xCenter + width / 2) * imageW;
      const yMax = (yCenter + height / 2) * imageH;

      // Extract class probabilities (assuming classes start at index 5)
      const classProbabilities = detection.slice(5, 85); // Assuming 80 classes.
      const classId = classProbabilities.indexOf(Math.max(...classProbabilities));

      // Confidence
      const confidence = detection[4];

      return {
        xMin: xMin,
        yMin: yMin,
        xMax: xMax,
        yMax: yMax,
        classId: classId,
        confidence: confidence,
      };
    });

    return results;
  }

  // Non-Maximum Suppression (NMS) Implementation (Simplified)
  function nonMaximumSuppression(detections, iouThreshold) {
    detections.sort((a, b) => b[4] - a[4]); // Sort by confidence (objectness score)

    const selected = [];
    const suppressed = new Set();

    for (let i = 0; i < detections.length; i++) {
      if (suppressed.has(i)) continue;

      selected.push(detections[i]);
      suppressed.add(i);

      for (let j = i + 1; j < detections.length; j++) {
        if (suppressed.has(j)) continue;

        const iou = calculateIOU(detections[i], detections[j]);
        if (iou > iouThreshold) {
          suppressed.add(j);
        }
      }
    }

    return selected;
  }

  // Calculate Intersection Over Union (IOU)
  function calculateIOU(box1, box2) {
    const x1Min = box1[0] - box1[2] / 2;
    const y1Min = box1[1] - box1[3] / 2;
    const x1Max = box1[0] + box1[2] / 2;
    const y1Max = box1[1] + box1[3] / 2;

    const x2Min = box2[0] - box2[2] / 2;
    const y2Min = box2[1] - box2[3] / 2;
    const x2Max = box2[0] + box2[2] / 2;
    const y2Max = box2[1] + box2[3] / 2;

    const xIntersectMin = Math.max(x1Min, x2Min);
    const yIntersectMin = Math.max(y1Min, y2Min);
    const xIntersectMax = Math.min(x1Max, x2Max);
    const yIntersectMax = Math.min(y1Max, y2Max);

    const intersectWidth = Math.max(0, xIntersectMax - xIntersectMin);
    const intersectHeight = Math.max(0, yIntersectMax - yIntersectMin);
    const intersectArea = intersectWidth * intersectHeight;

    const box1Area = (x1Max - x1Min) * (y1Max - y1Min);
    const box2Area = (x2Max - x2Min) * (y2Max - y2Min);
    const unionArea = box1Area + box2Area - intersectArea;

    return intersectArea / unionArea;
  }

  const processImage = async snapshot => {
    console.log('ProcessImage', snapshot);
    if (!snapshot) {
      console.log('No snapshot found', snapshot);
      return;
    }
    if (model == null) {
      console.error('No model found', model);
      return;
    }

    try {
      // 1. Resize the image if needed
      // 1. Convert the snapshot to the expected format
      console.log('Start convert snapshot to tensor');
      const inputTensor = await convertSnapshotToTensor(snapshot);
      if (!inputTensor) {
        console.error('Input tensor is undefined');
        return;
      }
      console.log('AI inputTensor:', inputTensor.length);

      // 2. Run model with given input buffer synchronously
      // console.log('Start model run');
      const outputs = await model.run([inputTensor]);
      console.log('Model run completed');
      if (!outputs) {
        console.error('Model run returned undefined outputs');
        return;
      }
      const detectedOut = outputs[0];
      if (!detectedOut) {
        console.error('Detected out is undefined');
        return;
      }
      console.log('AI detectedOut:', typeof detectedOut, detectedOut.length);

      console.log('AI model run:', detectedOut.length);
      const parsedYoloModels = parseYOLOOutput(detectedOut, 0.5, 0.4);
      console.log('AI parsedYoloModels:', parsedYoloModels.length);
      console.log('AI parsedYoloModels:', parsedYoloModels);
        setFoundObjects(parsedYoloModels);

      Object.values(parsedYoloModels).forEach((model) => {
        console.log('AI model:', classLabels[model.classId], model);
      })

      // 2. Run model with given input buffer synchronously
      // const outputs = model.runSync([inputTensor])

      // 3. Interpret outputs accordingly
      // const detection_boxes = outputs[0]
      // const detection_classes = outputs[1]
      // const detection_scores = outputs[2]
      // const num_detections = outputs[3]
      // console.log(`Detected ${num_detections[0]} objects!`)
      //
      // for (let i = 0; i < detection_boxes.length; i += 4) {
      //   const confidence = detection_scores[i / 4]
      //   if (confidence > 0.4) {
      //     // 4. Draw a red box around the detected object!
      //     const left = detection_boxes[i]
      //     const top = detection_boxes[i + 1]
      //     const right = detection_boxes[i + 2]
      //     const bottom = detection_boxes[i + 3]
      //     // const rect = SkRect.Make(left, top, right, bottom)
      //     // canvas.drawRect(rect, SkColors.Red)
      //     console.log('Detected object:', left, top, right, bottom, detection_classes[i / 4], confidence)
      //   }
      // }

      // 3. Interpret outputs accordinglyr
      // const detection_boxes = outputs[0];
      // const detection_classes = outputs[1];
      // const detection_scores = outputs[2];
      // const num_detections = outputs[3];
      // console.log(`Detected ${num_detections[0]} objects!`);
      //
      // for (let i = 0; i < detection_boxes.length; i += 4) {
      //   const confidence = detection_scores[i / 4];
      //   if (confidence > 0.7) {
      //     // 4. Draw a red box around the detected object!
      //     const left = detection_boxes[i];
      //     const top = detection_boxes[i + 1];
      //     const right = detection_boxes[i + 2];
      //     const bottom = detection_boxes[i + 3];
      //     // const rect = SkRect.Make(left, top, right, bottom);
      //     // canvas.drawRect(rect, SkColors.Red);
      //   }
      // }
    } catch (error) {
      console.error('Error running model:', error);
    }
  };

  // const frameProcessor = useFrameProcessor(
  //     (frame) => {
  //         'worklet'
  //         if (model == null) return
  //
  //         // 1. Resize 4k Frame to 192x192x3 using vision-camera-resize-plugin
  //         const resized = frame
  //         /*resize(frame, {
  //             scale: {
  //                 width: 192,
  //                 height: 192,
  //             },
  //             pixelFormat: 'rgb',
  //             dataType: 'uint8',
  //         })*/
  //
  //         // 2. Run model with given input buffer synchronously
  //         const outputs = model.runSync(resized)
  //
  //         // 3. Interpret outputs accordingly
  //         const detection_boxes = outputs[0]
  //         const detection_classes = outputs[1]
  //         const detection_scores = outputs[2]
  //         const num_detections = outputs[3]
  //         console.log(`Detected ${num_detections[0]} objects!`)
  //
  //         for (let i = 0; i < detection_boxes.length; i += 4) {
  //             const confidence = detection_scores[i / 4]
  //             if (confidence > 0.7) {
  //                 // 4. Draw a red box around the detected object!
  //                 const left = detection_boxes[i]
  //                 const top = detection_boxes[i + 1]
  //                 const right = detection_boxes[i + 2]
  //                 const bottom = detection_boxes[i + 3]
  //                 // const rect = SkRect.Make(left, top, right, bottom)
  //                 // canvas.drawRect(rect, SkColors.Red)
  //             }
  //         }
  //     },
  //     [model]
  // )

  useEffect(() => {
    const requestCameraPermission = async () => {
      const status = await Camera.requestCameraPermission();
      if (status === 'denied') {
        console.error('Camera permission denied');
      }
    };

    requestCameraPermission();
  }, []);

  if (device == null) {
    return (
      <View style={styles.loadingContainer}>
        <Text>No device...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        pixelFormat={'rgb'}
        ref={camera}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        photo={true}
      />
      <Button title="Capture" onPress={handleCapture} />
      <Button title="Process" onPress={() => processImage(image)} />
      {image && <Text>Image captured and processed!</Text>}
      <View style={{width: imageW , height: imageH , backgroundColor: '#ff000050'}}>
      {image && (
        <Image
          source={{uri: `file://${image.path}`}}
          style={{width: imageW, height: imageH, position: 'absolute', opacity: 0.5}}
        />
      )}
        {foundObjects.map((model, index) => {
            return (
                <View
                key={index}
                style={{
                    position: 'absolute',
                    left: model.xMin,
                    top: model.yMin,
                    width: model.xMax - model.xMin,
                    height: model.yMax - model.yMin,
                    borderWidth: 2,
                    borderColor: 'red',
                }}
                >
                <Text style={{color: 'red', fontSize: 12, fontWeight: 'bold'}}>
                    {classLabels[model.classId]}
                </Text>
                </View>
            );
        })}

      </View>
      {imageConvert && (
        <Image source={{uri: imageConvert}} style={{width: imageW, height: imageH}} />
      )}


    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'red',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  absoluteFill: {
    position: 'absolute',
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
  },
});

export default CameraDetect;
