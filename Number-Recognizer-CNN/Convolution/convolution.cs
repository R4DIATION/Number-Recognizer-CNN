using Number_Recognizer_CNN.Neural_Network;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Number_Recognizer_CNN.Convolution
{
    public class convolution
    {
        private double[,] picture;
        private int ActualValue;
        private NeuralNetwork NeuralNetwork;
        private List<double[,]> Filters = new List<double[,]>()
        {
            new double[,]
            {
                { 0, 200, 200, 200, 0 },
                { 0, 200,   0, 200, 0 },
                { 0, 200,   0, 200, 0 },
                { 0, 200,   0, 200, 0 },
                { 0, 200, 200, 200, 0 }
            },
            new double[,]
            {
                { 0,   0,   0,   0,   0 },
                { 0,   0,   0,   0,   0 },
                { 0,   0,   0,   0,   0 },
                { 0, 200,   0,   0, 200 },
                { 0,   0, 200, 200,   0 }
            },
            new double[,]
            {
                { 0, 0, 200, 0, 0 },
                { 0, 0, 200, 0, 0 },
                { 0, 0, 200, 0, 0 },
                { 0, 0, 200, 0, 0 },
                { 0, 0, 200, 0, 0 }
            },
            new double[,]
            {
                { 0, 200, 200, 200, 200 },
                { 0, 200,   0,  0,    0 },
                { 0, 200,   0,  0,    0 },
                { 0, 200,   0,  0,    0 },
                { 0, 200,   0,  0,    0 }
            },
            new double[,]
            {
                { 0,   0,   0, 200, 0 },
                { 0,   0, 200, 200, 0 },
                { 0, 200,   0, 200, 0 },
                { 0,   0,   0, 200, 0 },
                { 0,   0,   0, 200, 0 }
            },
            new double[,]
            {
                {   0,   0,   0, 200, 0 },
                {   0,   0, 200,   0, 0 },
                {   0, 200,   0,   0, 0 },
                { 200,   0,   0,   0, 0 },
                {   0,   0,   0,   0, 0 }
            },
            new double[,]
            {
                { 0,    0, 200, 200,   0 },
                { 0,  200,   0,   0, 200 },
                { 0,    0,   0,   0,   0 },
                { 0,    0,   0,   0,   0 },
                { 0,    0,   0,   0,   0 }
            }

        };
        private List<double[,]> FeatureMaps = new List<double[,]>();
        private List<double[,]> PooledImages = new List<double[,]>();
        private List<double[]> FlattenedImages = new List<double[]>();
        private int FilterIndex;
        public convolution(string imagepath, int actualValue, NeuralNetwork network)
        {
            this.NeuralNetwork = network;
            this.ActualValue = actualValue;
            this.picture = new double[28,28];
            Bitmap img = new Bitmap(imagepath);
            for (int i = 0; i < img.Width; i++)
            {
                for (int j = 0; j < img.Height; j++)
                {
                    Color pixel = img.GetPixel(j,i);
                    picture[i,j] = Math.Round(0.3 * pixel.R + 0.59 * pixel.G + 0.11 * pixel.B / 255) ;
                    Console.Write(" "+picture[i, j]);
                }
                Console.WriteLine();
            }
            img.Dispose();

            for (FilterIndex = 0; FilterIndex < Filters.Count; FilterIndex++)
            {
                Filtering();
                Pooling();
                Flattening();
            }

            double[] inputNeurons = new double[FlattenedImages.Sum(x=>x.Length)];
            int index = 0;
            for (int i = 0; i < FlattenedImages.Count; i++)
            {
                for (int j = 0; j < FlattenedImages[i].Length; j++)
                {
                    inputNeurons[index] = FlattenedImages[i][j];
                    index++;
                }
            }
            
            
            NeuralNetwork.InitNeuralNetwork(inputNeurons, inputNeurons.Length, this.ActualValue);
            

        }
        private void Flattening()
        {
            double[] inputNeurons = new double[PooledImages.First().GetLength(1)* PooledImages.First().GetLength(0)];
            int inputNeuronsIndex = 0;
            for (int i = 0; i < PooledImages[FilterIndex].GetLength(0); i++)
            {
                for (int j = 0; j < PooledImages[FilterIndex].GetLength(1); j++)
                {
                    inputNeurons[inputNeuronsIndex] = PooledImages[FilterIndex][i,j];
                    inputNeuronsIndex++;
                }
            }
            FlattenedImages.Add(inputNeurons);
        }
        private void Pooling()
        {
            double[,] pooledImage = new double[8,8];
            // pool size = 3*3
            for (int i = 0; i < pooledImage.GetLength(0); i++)
            {
                for (int j = 0; j < pooledImage.GetLength(1); j++)
                {
                    pooledImage[i, j] = PoolImage(i, j);
                }
            }
            PooledImages.Add(pooledImage);
        }
        private int PoolImage(int starting_i, int starting_j)
        {
            List<double> helper = new List<double>();
            for (int i = 0; i < 3; i++)
            {
                for (int j = 0; j < 3; j++)
                {
                    helper.Add(FeatureMaps[FilterIndex][starting_i + i, starting_j + j]);
                }
            }
            return (int)helper.Max();
        }
        private void Filtering()
        {
            double[,] featureMap = new double[24, 24];

            for (int i = 0; i < featureMap.GetLength(0); i++)
            {
                for (int j = 0; j < featureMap.GetLength(1); j++)
                {
                    featureMap[i, j] = FilterImage(i, j);
                }
            }

            FeatureMaps.Add(featureMap);
        }
        private int FilterImage(int startingPosInRow, int startingPosInLine)
        {
            int counter = 0;
            for (int i = 0; i < Filters[FilterIndex].GetLength(0); i++)
            {
                for (int j = 0; j < Filters[FilterIndex].GetLength(1); j++)
                {
                    if (picture[startingPosInRow + i, startingPosInLine + j] > Filters[FilterIndex][i,j])
                    {
                        counter++;
                    }
                }
            }
            return counter;
            
        }
    }
}
