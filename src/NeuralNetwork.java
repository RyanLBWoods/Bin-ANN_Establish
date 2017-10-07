import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;

public class NeuralNetwork {
    
    final static double[] Alex = {1, 0, 0, 0, 0};
    final static double[] Alfred = {0, 1, 0, 0, 0};
    final static double[] Anita = {0, 0, 1, 0, 0};
    final static double[] Anne = {0, 0, 0, 1, 0};
    final static double[] Bernard = {0, 0, 0, 0, 1};
    static BasicNetwork network = new BasicNetwork();
    static double INPUT[][] = new double[35][7];
    static double OUTPUT[][] = {Alex, Alfred, Anita, Anne, Bernard,
                Alex, Alfred, Anita, Anne, Bernard,
                Alex, Alfred, Anita, Anne, Bernard,
                Alex, Alfred, Anita, Anne, Bernard,
                Alex, Alfred, Anita, Anne, Bernard,
                Alex, Alfred, Anita, Anne, Bernard,
                Alex, Alfred, Anita, Anne, Bernard};
    
    public static void main(String[] args) {
        // TODO Auto-generated method stub
        String[] dataset = GetDataset.getData();
        
        for(int i = 0; i < dataset.length; i++){
            String[] chr = dataset[i].split(",");
            for(int j = 0; j < chr.length - 1; j++){
                if(chr[j].equals("No")){
                    chr[j] = "0";
                } else if(chr[j].equals("Yes")){
                    chr[j] = "1";
                }
                INPUT[i][j] = Integer.valueOf(chr[j]);
            }
        }
        
        trainNeuralNet();

    }
    
    private static void trainNeuralNet(){
        MLDataSet trainingSet = new BasicMLDataSet(INPUT, OUTPUT);
        
        int input_units = 7;
        int hidden_units = 13;
        int output_units = 5;
        
        
  
        network.addLayer(new BasicLayer(null, false, input_units));
        
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, hidden_units));
          
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, output_units));
          
        network.getStructure().finalizeStructure();
        network.reset();
        
        Backpropagation train = new Backpropagation(network, trainingSet, 0.025, 0.1);        
        
        int epoch = 1;
        do {
            train.iteration();
            System.out.println(/*"Epoch #" + epoch + " Error: " + */train.getError());
            epoch++;
            
        } while(train.getError() > 0.01);
        train.finishTraining();
    }
    
}