/**
 * @description: 线性回归
 * @author: Ashsay
 */
import * as tfvis from '@tensorflow/tfjs-vis';
import * as tf from '@tensorflow/tfjs';

window.onload = async() => {

  //初始化数据
  const xs = [1,2,3,4];
  const ys = [1,3,5,7];

  //散点图
  tfvis.render.scatterplot(
    { name: '线性回归训练集' },
    { values: xs.map((x, i) => ({x, y: ys[i]})) },
    { xAxisDomain: [0, 5], yAxisDomain: [0, 8] }
  )

  //创建连续模型
  const model = tf.sequential();
  //units: 一个神经元  inputShape: 特征数量
  model.add(tf.layers.dense({units:1, inputShape:[1]}))
  model.compile({ 
    //设置损失函数
    loss: tf.losses.meanSquaredError,
    //设置优化器  学习率
    optimizer: tf.train.sgd(0.1)
  });

  const inputs = tf.tensor(xs);
  const labels = tf.tensor(ys);
  await model.fit(inputs, labels, {
    //所需要学习的样本数据量
    batchSize: 4,
    //迭代次数
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks(
      { name: "训练过程" },
      ['loss']
    )
  });

  //预测结果
  const output = model.predict(tf.tensor([5]));
  output.print();
  console.log(output.dataSync()[0]);
  document.getElementById("result").innerHTML = output.dataSync()[0];

}