package priv.weilinwu.graphicmodel;

import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;

public class Main 
{
	public static final Logger logger = LoggerFactory.getLogger(Main.class);
	
	private final static String RESOURCE_PATH = "src/main/resources/";
	private final static String[] DATA_FILE_NAMES = {"classify_d3_k2_saved1.mat",
			"classify_d3_k2_saved2.mat", "classify_d3_k2_saved3.mat",
			"classify_d4_k3_saved1.mat", "classify_d4_k3_saved2.mat",
			"classify_d5_k3_saved1.mat", "classify_d5_k3_saved2.mat",
			"classify_d99_k50_saved1.mat", "classify_d99_k50_saved2.mat",
			"classify_d99_k60_saved1.mat", "classify_d99_k60_saved2.mat"};
	
    public static void main( String[] args ) throws IOException
    {
//    	for(int i = 0; i < DATA_FILE_NAMES.length; i++) {
//        	Utils.getTrainingSet(Utils.dataLoader(RESOURCE_PATH + DATA_FILE_NAMES[i]));
//        	Utils.getTestingSet(Utils.dataLoader(RESOURCE_PATH + DATA_FILE_NAMES[i]));
//    	}
//    	

    	Matrix[] trainingSet = Utils.getTrainingSet(Utils.dataLoader(RESOURCE_PATH + DATA_FILE_NAMES[0]));
    	Matrix[] testingSet = Utils.getTestingSet(Utils.dataLoader(RESOURCE_PATH + DATA_FILE_NAMES[0]));
    	
    	LogisticClassifier logisticClassifier = new LogisticClassifier(trainingSet, testingSet);
    	
//    	Matrix a = DenseMatrix.Factory.ones(4, 1);
//    	Matrix b = DenseMatrix.Factory.ones(4, 1);
//    	
//    	logisticClassifier.getValueOfLogisticFunction(a, b);
    	
    	logisticClassifier.getGradientDescendDirection();
    }
}
