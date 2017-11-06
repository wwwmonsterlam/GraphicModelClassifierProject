package priv.weilinwu.graphicmodel;

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
	private final double tolerance = 0.0001;
	
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
	
	// derivative of the objective function
	public Matrix getGradientDescendDirection() {
		Matrix sum = DenseMatrix.Factory.zeros(featureCount + 1, 1);
		
		for(int z = 0; z <= 1; z++) {
			for(int i = 0; i < trainingSampleCount; i++) {
				// Get the i th sample
				Matrix xi = trainingSet[z].subMatrix(Ret.NEW, 0, i, featureCount, i);
				sum = sum.plus(xi.times(z - getValueOfLogisticFunction(theta, xi)));
				logger.debug(xi.times(z - getValueOfLogisticFunction(theta, xi)).toString());
			}
		}
		
		logger.debug(sum.toString());
		
		return sum;
	}
	
	public double getValueOfLogisticFunction(Matrix theta, Matrix x) {
		double productOfThetaAndX = theta.transpose().mtimes(x).getAsDouble(0, 0);
		double value = 1.0 / (1 + Math.exp(-1 * productOfThetaAndX));
		
//		logger.debug(theta.toString());
//		logger.debug(theta.transpose().toString());
		logger.debug("product Of Theta And X is " + productOfThetaAndX);
		logger.debug("the value is " + value);
		
		return value;
	}
	
	// get (local) optimal step size
	public double getStepSize(Matrix gradientDescendDirection) {
		double stepSize = 0.0;
		
		return stepSize;
	}
	
	// return true if theta is updated, return false if the difference is within tolerance
	public boolean updateTheta(Matrix gradientDescendDirection) {
		double stepSize = getStepSize(gradientDescendDirection);
		Matrix difference = gradientDescendDirection.times(stepSize);
		if(isWithinTolerance(difference)) {
			return false;
		} else {
			theta = theta.plus(gradientDescendDirection.times(stepSize));
			return true;
		}
	}
	
	// return true if every entry is no larger than the tolerance value
	public boolean isWithinTolerance(Matrix m) {
		for(int i = 0; i <= featureCount; i++) {
			if(m.getAsDouble(i, 0) > tolerance) {
				return false;
			}
		}
		return true;
	}
	
}
