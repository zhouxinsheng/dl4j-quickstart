package org.deeplearning4j;

import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;

public class RestoreModel {

    public static void main(String... args){
        String kerasModelPath = "model.h5";
        ComputationGraph model = null;
        try{
            model = KerasModelImport.importKerasModelAndWeights(kerasModelPath);
        }catch (Exception e) {
            System.out.println("Error: import keras model failed!");
        }

        // test model target is 0.36568087339401245
        double[] samples = {0,1,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,13,159,4,2};
        double[] temp = new double[1];
        INDArray[] input = new INDArray[samples.length];
        for(int i=0; i!=samples.length; i++){
            temp[0] = samples[i];
            input[i] = Nd4j.create(temp);
        }
        INDArray output = model.outputSingle(input);
        System.out.println("**** output:" + output + "target: 0.36568087339401245");

        // test write model
        String savePath = "model.zip";
        try{
            ModelSerializer.writeModel(model, savePath, true);
            //model.save(new File(savePath));
        } catch (Exception e) {
            System.out.println("Error: save model failed!");
        }

        // test restore model
        try{
            model = ModelSerializer.restoreComputationGraph(savePath);
        } catch (Exception e) {
            System.out.println(e);
            System.out.println("Error: restore model failed!");
            return;
        }

        INDArray output2 = model.outputSingle(input);
        System.out.println("#### output:" + output2 + "target: 0.36568087339401245");
    }
}