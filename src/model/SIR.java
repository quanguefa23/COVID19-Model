package model;

import entity.Country;
import entity.DataEachDay;
import getdata.ReadDataFromCSV;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class SIR {
    private static final int START_DAY_FOR_TRAIN = 100;
    private static final int END_DAY_FOR_TRAIN = 130;

    private static final int START_DAY_FOR_TEST = 100;
    private static final int END_DAY_FOR_TEST = 130;

    public static void main(String... argv) {
        ReadDataFromCSV readDataFromCSV = new ReadDataFromCSV();
        Map<String, Country> dataCountries = readDataFromCSV.getDataCountries();

        // sample data to run model: Trump
        Country us = dataCountries.get("us");

        // train model (calculate params)
        SIRParam param = trainModelDerivativeMethod(us, START_DAY_FOR_TRAIN, END_DAY_FOR_TRAIN);

        // run model
        List<DataEachDay> resultRunModel = runModel(us.getDataSet().get(START_DAY_FOR_TEST), param.beta, param.gamma,
                us.getN(), END_DAY_FOR_TEST - START_DAY_FOR_TEST);

        // print result to console
        printResult(resultRunModel, us);
    }

    private static void printResult(List<DataEachDay> resultRunModel, Country us) {

        int i = START_DAY_FOR_TEST;
        for(DataEachDay dayTrain : resultRunModel) {
            DataEachDay dayRealValue = us.getDataSet().get(++i);
            System.out.println("---- Day " + i + " ----");
            System.out.println("Train result: " + dayTrain.S + " | " + dayTrain.I + " | " + dayTrain.R);
            System.out.println("Real value: " + dayRealValue.S + " | " + dayRealValue.I + " | " + dayRealValue.R);
            System.out.println("Relative delta: " +
                    100.0 * Math.abs(dayTrain.S - dayRealValue.S) / dayRealValue.S + "% | " +
                    100.0 * Math.abs(dayTrain.I - dayRealValue.I) / dayRealValue.I + "% | " +
                    100.0 * Math.abs(dayTrain.R - dayRealValue.R) / dayRealValue.R + "%");
        }
    }

    private static List<DataEachDay> runModel(DataEachDay genesisDay, double beta, double gamma, long N, int period) {
        List<DataEachDay> list = new ArrayList<>();

        DataEachDay yesterday = genesisDay;
        for (int i = 0; i < period; i++) {
            long St = yesterday.S;
            long It = yesterday.I;
            long Rt = yesterday.R;

            // calculate new data
            long St_1 = (long) (St - beta * St * It / N);
            long It_1 = (long) (It + beta * St * It / N - gamma * It);
            long Rt_1 = (long) (Rt + gamma * It);

            DataEachDay newDay = new DataEachDay(St_1, It_1, Rt_1);
            list.add(newDay);
            yesterday = newDay;
        }

        return list;
    }

    private static SIRParam trainModelDerivativeMethod(Country country, int startDay, int endDay) {
        double numBeta = 0, denBeta = 0;
        long numGamma = 0, denGamma = 0;

        for(int i = startDay; i < endDay; i++) {
            DataEachDay d1 = country.getDataSet().get(i);
            DataEachDay d2 = country.getDataSet().get(i + 1);

            long N = country.getN();
            long St = d1.S;
            long St_1 = d2.S; // S(t+1)
            long It = d1.I;
            long Rt = d1.R;
            long Rt_1 = d2.R; // R(t+1)
            double At = 1.0 * St * It / N;

            numBeta += At * (St - St_1);
            denBeta += At * At;

            numGamma += It * (Rt_1 - Rt);
            denGamma += It * It;
        }

        double beta = numBeta / denBeta;
        double gamma = 1.0 * numGamma / denGamma;

        return new SIRParam(beta, gamma);
    }

    private static SIRParam trainModelAverageMethod(Country country, int startDay, int endDay) {
        List<SIRParam> listParams = new ArrayList<>();

        for(int i = startDay; i < endDay; i++) {
            DataEachDay d1 = country.getDataSet().get(i);
            DataEachDay d2 = country.getDataSet().get(i + 1);

            long N = country.getN();
            long St = d1.S;
            long St_1 = d2.S; // S(t+1)
            long It = d1.I;
            long Rt = d1.R;
            long Rt_1 = d2.R; // R(t+1)

            SIRParam param = new SIRParam();
            param.beta = 1.0 * (St - St_1) * N / (St * It);
            param.gamma = 1.0 * (Rt_1 - Rt) / It;

            listParams.add(param);
        }

        SIRParam res = new SIRParam();
        for(SIRParam param : listParams) {
            res.beta += param.beta;
            res.gamma += param.gamma;
        }

        res.beta /= listParams.size();
        res.gamma /= listParams.size();

        return res;
    }

    static class SIRParam {
        public double beta;
        public double gamma;

        public SIRParam(double beta, double gamma) {
            this.beta = beta;
            this.gamma = gamma;
        }

        public SIRParam() {
            this.beta = 0;
            this.gamma = 0;
        }
    }
}
