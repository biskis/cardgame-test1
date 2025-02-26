import React, {useEffect, useRef, useState} from 'react';
import {StyleSheet, View, Text, Button, Image} from 'react-native';
import {useTensorflowModel} from 'react-native-fast-tflite';
import {convertToRGB} from 'react-native-image-to-rgb';
import exampleImage from '../../assets/example.jpg'



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
  const objectDetection = useTensorflowModel(require('../../assets/ai/aimodel.tflite'));
  const model = objectDetection.state === 'loaded' ? objectDetection.model : undefined;
  const [foundObjects, setFoundObjects] = useState([]);



  const convertSnapshotToTensor = async () => {
    const imageUri = Image.resolveAssetSource(exampleImage).uri
    // // const imageUri = 'file:///data/user/0/com.cardgame/cache/be8e1073-64e0-4ada-a91b-ef2cff09a447.JPEG'

    const convertedArray = await convertToRGB(imageUri);
    const array = new Float32Array(convertedArray);

    return array
  };

  const processImage = async () => {
    if (model == null) {
      console.error('No model found', model);
      return;
    }

    try {
      // 1. Resize the image if needed
      // 1. Convert the snapshot to the expected format
      console.log('Start convert snapshot to tensor');
      const inputTensor = await convertSnapshotToTensor();
      console.log('AI inputTensor:', inputTensor.length); // 1228800

      if (!inputTensor) {
        throw new Error('Input tensor is undefined');
      }

      const outputs = await model.run([inputTensor]);
      const detectedOut = outputs[0];
      console.log('AI detectedOut:', typeof detectedOut, detectedOut.length); //470400


      const parsedYoloModels = parseYOLOOutput(detectedOut, 0.5, 0.4);
      console.log('AI parsedYoloModels:', parsedYoloModels.length);
      console.log('AI parsedYoloModels:', parsedYoloModels);
      setFoundObjects(parsedYoloModels);

      Object.values(parsedYoloModels).forEach((model) => {
        console.log('AI model:', classLabels[model.classId], model);
      })

    } catch (error) {
      console.error('Error running model:', error);
    }
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
      const xc = outputTensor[i];
      const yc = outputTensor[i + numDetections];
      const w = outputTensor[i + numDetections * 2];
      const h = outputTensor[i + numDetections * 3];
      // also use here all the channels: numChannels
      const classes = []
      for(let j = 4; j < numChannels; j++) {
        classes.push(outputTensor[i + numDetections * j]);
      }

      const confidence = Math.max(...classes);

      detections.push({
        xc,
        yc,
        w,
        h,
        confidence,
        classes
      });
    }

    // Filter detections based on confidence
    const filteredDetections = detections.filter(detection => detection.confidence > confidenceThreshold);

    console.log('filteredDetections', filteredDetections.length)

    // Non-Maximum Suppression (NMS)
    // const nmsDetections = nonMaximumSuppression(filteredDetections, iouThreshold);

    // Parse and format the results
    const results = filteredDetections.map(detection => {
      // Extract bounding box coordinates
      const xCenter = detection.xc;
      const yCenter = detection.yc;
      const width = detection.w;
      const height = detection.h;

      const xMin = (xCenter - width / 2) * imageW;
      const yMin = (yCenter - height / 2) * imageH;
      const xMax = (xCenter + width / 2) * imageW;
      const yMax = (yCenter + height / 2) * imageH;

      // Extract class probabilities (assuming classes start at index 5)
      const classProbabilities = detection.classes
      const classId = classProbabilities.indexOf(Math.max(...classProbabilities));

      // Confidence
      const confidence = detection.confidence

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
  /*function nonMaximumSuppression(detections, iouThreshold) {
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
  }*/

  return (
    <View style={styles.container}>
      <Button title="Process" onPress={processImage} />

      <View style={{width: imageW , height: imageH , backgroundColor: '#ff000010'}}>

        <Image
          source={exampleImage}
          style={{width: imageW, height: imageH, position: 'absolute', opacity: 0.5}}
        />

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
