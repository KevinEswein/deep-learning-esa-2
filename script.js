/**
 * Ground-Truth
 */
function functionY(x) {
  return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

/**
 * ------ 1. Generate input data ------
 */
function generateData(N, noiseVar) {
  const xValues = [];
  const yValues = [];
  const noise = (mean, variance) => mean + Math.sqrt(variance) * tf.randomNormal([1]).dataSync()[0];

  for (let i = 0; i < N; i++) {
    const x = Math.random() * 4 - 2; // Evenly distributed x-values in the interval [-2, +2]
    const y = functionY(x);
    xValues.push(x);
    yValues.push(y);
  }

  // Data without noise
  const dataWithoutNoise = { x: xValues, y: yValues };

  // Data with noise
  const yValuesNoisy = yValues.map(y => y + noise(0, noiseVar));
  const dataWithNoise = { x: xValues, y: yValuesNoisy };

  return { dataWithoutNoise, dataWithNoise };
}

let dataWithoutNoise, dataWithNoise;
let trainDataWithoutNoise, testDataWithoutNoise;
let trainDataWithNoise, testDataWithNoise;

function generateAndSplitData() {
  const dataSize = parseInt(document.getElementById('dataSize').value);
  const noiseVariance = parseFloat(document.getElementById('noiseVariance').value);

  ({ dataWithoutNoise, dataWithNoise } = generateData(dataSize, noiseVariance));
  ({ trainData: trainDataWithoutNoise, testData: testDataWithoutNoise } = splitData(dataWithoutNoise, 0.5));
  ({ trainData: trainDataWithNoise, testData: testDataWithNoise } = splitData(dataWithNoise, 0.5));

  // Visualize the data
  plotData('plot-dataset', trainDataWithoutNoise, testDataWithoutNoise, 'red', 'blue', '1.1 Unverrauschte Daten');
  plotData('plot-dataset-noisy', trainDataWithNoise, testDataWithNoise, 'red', 'blue', '1.2 Verrauschte Daten');
}

function saveData() {
  const data = {
    dataWithoutNoise,
    dataWithNoise,
    trainDataWithoutNoise,
    testDataWithoutNoise,
    trainDataWithNoise,
    testDataWithNoise
  };
  localStorage.setItem('data', JSON.stringify(data));
}

function loadData() {
  const data = JSON.parse(localStorage.getItem('data'));
  if (data) {
    dataWithoutNoise = data.dataWithoutNoise;
    dataWithNoise = data.dataWithNoise;
    trainDataWithoutNoise = data.trainDataWithoutNoise;
    testDataWithoutNoise = data.testDataWithoutNoise;
    trainDataWithNoise = data.trainDataWithNoise;
    testDataWithNoise = data.testDataWithNoise;

    // Visualize the data
    plotData('plot-dataset', trainDataWithoutNoise, testDataWithoutNoise, 'red', 'blue', '1.1 Unverrauschte Daten');
    plotData('plot-dataset-noisy', trainDataWithNoise, testDataWithNoise, 'red', 'blue', '1.2 Verrauschte Daten');
  } else {
    alert('Keine gespeicherten Daten gefunden.');
  }
}

/**
 * ------ 2. Split data (training / testing) ------
 */
function splitData(data, trainRatio) {
  const N = data.x.length;
  const trainSize = Math.floor(N * trainRatio);
  const indices = Array.from({ length: N }, (_, i) => i);

  // Random permutation of the indices
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }

  const trainIndices = indices.slice(0, trainSize);
  const testIndices = indices.slice(trainSize);

  const trainData = {
    x: trainIndices.map(i => data.x[i]),
    y: trainIndices.map(i => data.y[i])
  };

  const testData = {
    x: testIndices.map(i => data.x[i]),
    y: testIndices.map(i => data.y[i])
  };

  return { trainData, testData };
}

/**
 * ------ 3. Create model ------
 */
function createModel(learningRate) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [1] }));
  model.add(tf.layers.dense({ units: 100, activation: 'relu' }));
  model.add(tf.layers.dense({ units: 1 })); // Output layer, linear activation (default)
  model.compile({ optimizer: tf.train.adam(learningRate), loss: 'meanSquaredError' });
  return model;
}

let model, modelBestFit, modelOverFit;

function initializeModels() {
  const learningRate = parseFloat(document.getElementById('learningRate').value);
  model = createModel(learningRate);
  modelBestFit = createModel(learningRate);
  modelOverFit = createModel(learningRate);

  // Save models in the window-object
  window['model'] = model;
  window['modelBestFit'] = modelBestFit;
  window['modelOverFit'] = modelOverFit;
}

/**
 * ------ 4. Train models ------
 */
async function trainModel(model, trainXs, trainYs, epochs ) {
  await model.fit(trainXs, trainYs, {
    epochs: epochs,
    validationSplit: 0.2,
    batchSize: 32,
    callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 10 })
  });
}

async function trainModels() {
  initializeModels();
  const epochs = parseInt(document.getElementById('epochs').value);

  // Prepare train data
  const trainXsUnverrauscht = tf.tensor2d(trainDataWithoutNoise.x, [trainDataWithoutNoise.x.length, 1]);
  const trainYsUnverrauscht = tf.tensor2d(trainDataWithoutNoise.y, [trainDataWithoutNoise.y.length, 1]);

  await trainModel(model, trainXsUnverrauscht, trainYsUnverrauscht, epochs).then(() => {
    // 1) Prepare test data
    const testXsUnverrauscht = tf.tensor2d(testDataWithoutNoise.x, [testDataWithoutNoise.x.length, 1]);
    const testYsUnverrauscht = tf.tensor2d(testDataWithoutNoise.y, [testDataWithoutNoise.y.length, 1]);

    // 2) Evaluate model
    const lossTrain = model.evaluate(trainXsUnverrauscht, trainYsUnverrauscht);
    const lossTest = model.evaluate(testXsUnverrauscht, testYsUnverrauscht);
    console.log('Train Loss (Unverrauscht):', Number(lossTrain.dataSync()[0].toPrecision(4)));
    console.log('Test Loss (Unverrauscht):', Number(lossTest.dataSync()[0].toPrecision(4)));

    // 3) Prognosis for visualization
    const predictionsTrain = model.predict(trainXsUnverrauscht).dataSync();
    const predictionsTest = model.predict(testXsUnverrauscht).dataSync();

    // 4) Visualization
    plotPredictions('plot-train', trainDataWithoutNoise, predictionsTrain, 'blue', 'orange', '2.1 Vorhersage auf Trainingsdaten (Unverrauscht)');
    plotPredictions('plot-test', testDataWithoutNoise, predictionsTest, 'red', 'orange', '2.2 Vorhersage auf Testdaten (Unverrauscht)');
    document.getElementById('loss-train').innerText = `Train Loss: ${Number(lossTrain.dataSync()[0].toPrecision(4))}`;
    document.getElementById('loss-test').innerText = `Test Loss: ${Number(lossTest.dataSync()[0].toPrecision(4))}`;
  });

  const trainXsVerrauscht = tf.tensor2d(trainDataWithNoise.x, [trainDataWithNoise.x.length, 1]);
  const trainYsVerrauscht = tf.tensor2d(trainDataWithNoise.y, [trainDataWithNoise.y.length, 1]);

  await trainModel(modelBestFit, trainXsVerrauscht, trainYsVerrauscht, epochs).then(() => {
    // 1) Prepare test data
    const testXsVerrauscht = tf.tensor2d(testDataWithNoise.x, [testDataWithNoise.x.length, 1]);
    const testYsVerrauscht = tf.tensor2d(testDataWithNoise.y, [testDataWithNoise.y.length, 1]);

    // 2) Evaluate model
    const lossTrain = modelBestFit.evaluate(trainXsVerrauscht, trainYsVerrauscht);
    const lossTest = modelBestFit.evaluate(testXsVerrauscht, testYsVerrauscht);
    console.log('Train Loss (BestFit):', Number(lossTrain.dataSync()[0].toPrecision(4)));
    console.log('Test Loss (BestFit):', Number(lossTest.dataSync()[0].toPrecision(4)));

    // 3) Prognosis for visualization
    const predictionsTrain = modelBestFit.predict(trainXsVerrauscht).dataSync();
    const predictionsTest = modelBestFit.predict(testXsVerrauscht).dataSync();

    // 4) Visualization
    plotPredictions('plot-train-best-fit', trainDataWithNoise, predictionsTrain, 'blue', 'orange', '3.1 Vorhersage auf Trainingsdaten (BestFit)');
    plotPredictions('plot-test-best-fit', testDataWithNoise, predictionsTest, 'red', 'orange', '3.2 Vorhersage auf Testdaten (BestFit)');
    document.getElementById('loss-train-best-fit').innerText = `Train Loss: ${Number(lossTrain.dataSync()[0].toPrecision(4))}`;
    document.getElementById('loss-test-best-fit').innerText = `Test Loss: ${Number(lossTest.dataSync()[0].toPrecision(4))}`;
  });

  await trainModel(modelOverFit, trainXsVerrauscht, trainYsVerrauscht, 500).then(() => {
    // 1) Prepare test data
    const testXsVerrauscht = tf.tensor2d(testDataWithNoise.x, [testDataWithNoise.x.length, 1]);
    const testYsVerrauscht = tf.tensor2d(testDataWithNoise.y, [testDataWithNoise.y.length, 1]);

    // 2) Evaluate model
    const lossTrain = modelOverFit.evaluate(trainXsVerrauscht, trainYsVerrauscht);
    const lossTest = modelOverFit.evaluate(testXsVerrauscht, testYsVerrauscht);
    console.log('Train Loss (OverFit):', Number(lossTrain.dataSync()[0].toPrecision(4)));
    console.log('Test Loss (OverFit):', Number(lossTest.dataSync()[0].toPrecision(4)));

    // 3) Prognosis for visualization
    const predictionsTrain = modelOverFit.predict(trainXsVerrauscht).dataSync();
    const predictionsTest = modelOverFit.predict(testXsVerrauscht).dataSync();

    // 4) Visualization
    plotPredictions('plot-train-over-fit', trainDataWithNoise, predictionsTrain, 'blue', 'orange', '4.1 Vorhersage auf Trainingsdaten (OverFit)');
    plotPredictions('plot-test-over-fit', testDataWithNoise, predictionsTest, 'red', 'orange', '4.2 Vorhersage auf Testdaten (OverFit)');
    document.getElementById('loss-train-over-fit').innerText = `Train Loss: ${Number(lossTrain.dataSync()[0].toPrecision(4))}`;
    document.getElementById('loss-test-over-fit').innerText = `Test Loss: ${Number(lossTest.dataSync()[0].toPrecision(4))}`;
  });
}

/**
 * ------ 5. Save and load models ------
 */
async function saveModel(modelName) {
  const model = window[modelName];
  if (!model) {
    alert(`Modell ${modelName} existiert nicht.`);
    return;
  }
  await model.save(`localstorage://${modelName}`);
  alert(`${modelName} gespeichert.`);
}

async function loadModel(modelName) {
  try {
    // Load model
    window[modelName] = await tf.loadLayersModel(`localstorage://${modelName}`);
    alert(`${modelName} geladen.`);

    // Visualize predictions
    const trainXsVerrauscht = tf.tensor2d(trainDataWithNoise.x, [trainDataWithNoise.x.length, 1]);
    const testXsVerrauscht = tf.tensor2d(testDataWithNoise.x, [testDataWithNoise.x.length, 1]);

    const predictionsTrain = window[modelName].predict(trainXsVerrauscht).dataSync();
    const predictionsTest = window[modelName].predict(testXsVerrauscht).dataSync();

    if (modelName === 'modelBestFit') {
      plotPredictions('plot-train-best-fit', trainDataWithNoise, predictionsTrain, 'blue', 'orange', '3.1 Vorhersage auf Trainingsdaten (BestFit)');
      plotPredictions('plot-test-best-fit', testDataWithNoise, predictionsTest, 'red', 'orange', '3.2 Vorhersage auf Testdaten (BestFit)');
    } else if (modelName === 'modelOverFit') {
      plotPredictions('plot-train-over-fit', trainDataWithNoise, predictionsTrain, 'blue', 'orange', '4.1 Vorhersage auf Trainingsdaten (OverFit)');
      plotPredictions('plot-test-over-fit', testDataWithNoise, predictionsTest, 'red', 'orange', '4.2 Vorhersage auf Testdaten (OverFit)');
    }
  } catch (error) {
    console.error(`Error loading model ${modelName}:`, error);
  }
}

/**
 * ------ 6. Plotting functions ------
 */
function plotData(plotId, trainData, testData, colorTest, colorTrain, title) {
  const traceTest = {
    x: testData.x,
    y: testData.y,
    mode: 'markers',
    type: 'scatter',
    name: 'Testdaten',
    marker: { color: colorTest }
  };

  const traceTrain = {
    x: trainData.x,
    y: trainData.y,
    mode: 'markers',
    type: 'scatter',
    name: 'Trainingsdaten',
    marker: { color: colorTrain }
  };

  const layout = {
    title: title,
    xaxis: { title: 'x' },
    yaxis: { title: 'y' }
  };

  Plotly.newPlot(plotId, [traceTrain, traceTest], layout);
}

function plotPredictions(plotId, data, predictions, colorData, colorPred, title) {
  // Sort data for prognosis
  const sortedIndices = data.x.map((_, i) => i).sort((a, b) => data.x[a] - data.x[b]);
  const sortedX = sortedIndices.map(i => data.x[i]);
  const sortedPredictions = sortedIndices.map(i => predictions[i]);

  const traceData = {
    x: data.x,
    y: data.y,
    mode: 'markers',
    type: 'scatter',
    name: 'Daten',
    marker: { color: colorData }
  };

  const tracePred = {
    x: sortedX,
    y: sortedPredictions,
    mode: 'lines',
    type: 'scatter',
    name: 'Vorhersage',
    line: { color: colorPred }
  };

  const layout = {
    title: title,
    xaxis: { title: 'x' },
    yaxis: { title: 'y' }
  };

  Plotly.newPlot(plotId, [traceData, tracePred], layout);
}

// Initial generation and training of models
generateAndSplitData();
initializeModels();
trainModels();
