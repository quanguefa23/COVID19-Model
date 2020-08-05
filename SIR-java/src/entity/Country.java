package entity;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class Country {
    private String name;
    private long N; // total population
    private List<DataEachDay> dataSet;

    public long getN() {
        return N;
    }

    public List<DataEachDay> getDataSet() {
        return dataSet;
    }

    public Country(String name, long N, File file) {
        this.name = name;
        this.N = N;
        this.dataSet = new ArrayList<>();

        readDataForCountry(file);
    }

    private void readDataForCountry(File file) {
        try {
            BufferedReader br = new BufferedReader(new FileReader(file));

            String line;
            String[] tempArr;
            br.readLine(); // skip first line
            while((line = br.readLine()) != null) {
                tempArr = line.split(",");

                long confirm = Long.parseLong(tempArr[0]);
                long death = Long.parseLong(tempArr[1]);
                long recover = Long.parseLong(tempArr[2]);

                DataEachDay dataEachDay = new DataEachDay(N, confirm, death, recover);
                dataSet.add(dataEachDay);
            }

            br.close();
        } catch (Exception e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }
}
