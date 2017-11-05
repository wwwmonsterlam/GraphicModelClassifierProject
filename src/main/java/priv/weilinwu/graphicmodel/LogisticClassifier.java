package priv.weilinwu.graphicmodel;

import javax.naming.spi.DirStateFactory.Result;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.ujmp.core.DenseMatrix;
import org.ujmp.core.Matrix;
import org.ujmp.core.calculation.Calculation.Ret;

public class LogisticClassifier {
	public static final Logger logger = LoggerFactory.getLogger(LogisticClassifier.class);
	
	private Matrix[] trainingSet;
	private Matrix[] testingSet;
	private Matrix theta;
	private long featureCount;
	private long trainingSampleCount;
	
	LogisticClassifier(Matrix[] trainingSet, Matrix[] testingSet) {
		// For calculation simplicity, add one to each column
		// In this way, theta can direct multiply each sample vector in the data set
		this.trainingSet = addOneToEachColumn(trainingSet);
		this.testingSet = addOneToEachColumn(testingSet);
		logger.debug(this.testingSet[0].toString());
		this.featureCount = trainingSet[0].getRowCount();
		this.trainingSampleCount = trainingSet[0].getColumnCount();
		theta = DenseMatrix.Factory.zeros(this.featureCount + 1 , 1);
		logger.debug(theta.toString());
	}
	
	private Matrix[] addOneToEachColumn(Matrix[] m) {
		Matrix temp = DenseMatrix.Factory.ones(1 , m[0].getColumnCount());
		Matrix modifiedClass0 = temp.appendVertically(Ret.NEW, m[0]);
		Matrix modifiedClass1 = temp.appendVertically(Ret.NEW, m[1]);
		Matrix[] result = {modifiedClass0, modifiedClass1};
		
		return result;
	}
	
}
