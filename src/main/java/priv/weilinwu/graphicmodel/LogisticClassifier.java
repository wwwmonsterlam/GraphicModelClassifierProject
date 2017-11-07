package priv.weilinwu.graphicmodel;

import java.io.ObjectInputStream.GetField;

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
	private Matrix gradientDescendDirection;
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
		
		generateNewGradientDescendDirection();
	}
	
	private Matrix[] addOneToEachColumn(Matrix[] m) {
		Matrix temp = DenseMatrix.Factory.ones(1 , m[0].getColumnCount());
		Matrix modifiedClass0 = temp.appendVertically(Ret.NEW, m[0]);
		Matrix modifiedClass1 = temp.appendVertically(Ret.NEW, m[1]);
		Matrix[] result = {modifiedClass0, modifiedClass1};
		
		return result;
	}
	
	// derivative of the objective function
	public void generateNewGradientDescendDirection() {
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
		
		this.gradientDescendDirection = sum;
	}
	
	public double getValueOfLogisticFunction(Matrix theta, Matrix x) {
		double productOfThetaAndX = theta.transpose().mtimes(x).getAsDouble(0, 0);
		double value = 1.0 / (1 + Math.exp(-1 * productOfThetaAndX));
		
		logger.debug("product Of Theta And X is " + productOfThetaAndX);
		logger.debug("the value is " + value);
		
		return value;
	}
	
	// get (local) optimal step size
	public double getOptimalStepSize() {
		double[] pointPair = getInitialPointPair();
		updatePointPairUntillDistanceWithinTolerance(pointPair[0], pointPair[1]);

		double stepSize = (pointPair[0] + pointPair[1]) / 2.0;
		return stepSize;
	}
	
	public double[] getInitialPointPair() {
		double stepSize = 0.1;
		double searchSize = 1.0;
		double left = 0;
		double right = 0;
		
		if(getValueOfObjectiveFunctionderivativeWithRespectToStepSize(stepSize) <= 0) {
			right = stepSize;
			left = right - searchSize;
			while(getValueOfObjectiveFunctionderivativeWithRespectToStepSize(left) <= 0) {
				searchSize *= 2;
				left = right - searchSize;
			}
		} else {
			left = stepSize;
			right = left + searchSize;
			while(getValueOfObjectiveFunctionderivativeWithRespectToStepSize(right) > 0) {
				searchSize *= 2;
				right = left + searchSize;
			}
		}
		
		double[] initialPointPaire = {left, right};
		return initialPointPaire;
	}
	
	public void updatePointPairUntillDistanceWithinTolerance(double left, double right) {
		if(right - left < tolerance) {
			return;
		}
		
		double middle = (right - left) / 2.0;
		if(getValueOfObjectiveFunctionderivativeWithRespectToStepSize(middle) <= 0) {
			right = middle;
		} else {
			left = middle;
		}
		
		updatePointPairUntillDistanceWithinTolerance(left, right);
	}
	
	// return true if theta is updated, return false if the difference is within tolerance
	public boolean updateTheta() {
		generateNewGradientDescendDirection();
		double stepSize = getOptimalStepSize();
//		double stepSize = 0.0000000001;
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
	
	// this is used to help find optimal step size
	public double getValueOfObjectiveFunctionderivativeWithRespectToStepSize(double stepSize) {
		double sum = 0;
		
		for(int z = 0; z <= 1; z++) {
			for(int i = 0; i < trainingSampleCount; i++) {
				// Get the i th sample
				Matrix xi = trainingSet[z].subMatrix(Ret.NEW, 0, i, featureCount, i);
				Matrix newTheta = theta.plus(gradientDescendDirection.times(stepSize));
				sum += gradientDescendDirection.transpose().times(xi).getAsDouble(0, 0) *
						(z - getValueOfLogisticFunction(newTheta, xi));
			}
		}
		
		return sum;
	}
	
	public Matrix getTrainedTheta() {
		while(updateTheta());
		
		return theta;
	}
}

