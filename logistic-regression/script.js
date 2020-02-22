/**
 * @description: 逻辑回归
 * @author: Ashsay
 */
import * as tfvis from '@tensorflow/tfjs-vis';
import * as tf from '@tensorflow/tfjs';
import { getData } from './data';

window.onload = async() => {

  const data = getData(400);
  console.log(data);

  tfvis.render.scatterplot(
    {name:"逻辑回归训练数据"},
    {
      values: [
        data.filter(p=>p.label == 1),
        data.filter(p=>p.label == 0)
      ]
    }
  )

  const model = tf.sequential();
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [2],
    //激活函数，将输出值压缩到 0-1 之间
    activation: 'sigmoid'
  }))
  model.compile(
    //对数损失函数
    { loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1) }
  )

  const inputs = tf.tensor(data.map(p=>[p.x, p.y]));
  const labels = tf.tensor(data.map(p=> p.label));

  await model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 10,
    callbacks: tfvis.show.fitCallbacks(
      { name: '训练过程' },
      ['loss']
    )
  })

  window.predict = (form) => {
    const pred = model.predict(tf.tensor([[form.x.value * 1, form.y.value * 1]]))
    alert(`预测结果为:${pred.dataSync()[0]}`)
  }
  
};

