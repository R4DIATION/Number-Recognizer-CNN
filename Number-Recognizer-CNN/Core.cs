using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Number_Recognizer_CNN
{
    public abstract class Core
    {
        public enum LayerType
        {
            None, 
            Input, 
            Output,
            Hidden
        }
        public enum ActivationType
        {
            None, 
            Sigmoid,
            ReLU
        }
        public static Random rnd = new Random();
        public static double Sigmoid(double num)
        {
            return (double)1 / (double)Math.Pow((double)Math.E, (double)-(num)); 
        }
        public static double Cost(double num, double expected)
        {
            return Math.Pow(num - expected, 2);
        }
    }
}
