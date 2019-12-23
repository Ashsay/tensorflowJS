/**
 * @description: 张量 --> 多维数组
 *               先把被点乘数组旋转后再运算 --> 横向改为纵向再做乘法
 *               [[1,2],[3,4],[5,6]]  -->  [[1,3,5],[2,4,6]]
 */
import * as tf from '@tensorflow/tfjs';

const t0 = tf.tensor(1);
t0.print();
console.log(t0);

const t1 = tf.tensor([1,2])
t1.print();
console.log(t1)

const t2 = tf.tensor([[1,2],[3,4]])
t2.print();
console.log(t2);

//传统的for循环
const input = [1,2,3,4];
const w = [[1,2,3,4],[2,3,4,5],[3,4,5,6],[4,5,6,7]];
const output = [0,0,0,0];

for(let i = 0; i < w.length; i ++){
  for(let j = 0; j < input.length; j ++){
    output[i] += input[j] * w[i][j]
  }
}

console.log(output);

//tensorflow点乘法
const dots = tf.tensor(w).dot(tf.tensor(input));
dots.print();
console.log(dots);

//点乘思路
const tpArr = [[1,2],[3,4],[5,6]];
tf.transpose(tpArr).print();
const cal = [1,2,3];
tf.tensor(cal).print();
tf.tensor(cal).dot(tf.tensor(tpArr)).print();

// 张量切割操作
const arrOfTensor2d = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
const d2 = 2;
//第一个为数组样本  [第一维度数量, 第二维度数量]
const tensor2dArr = tf.tensor2d(arrOfTensor2d, [arrOfTensor2d.length/d2, d2]);
tensor2dArr.print();
//第一个位起点， 第二个位大小 [竖向数量，横向数量]
const result2d = tensor2dArr.slice([2,0], [2,1]);
result2d.print();

//数据随机化  洗牌操作
const arrOfShuffle = [1, 2, 3, 4, 5, 6, 7];
//会改变原来的数组
tf.util.shuffle(arrOfShuffle);
console.log(arrOfShuffle);