package getdata;

import entity.Country;

import java.io.*;
import java.util.*;

public class ReadDataFromCSV {

    private final String POPULATION_FILE_URL = "./data/population.csv";
    private final String DATA_COUNTRIES_FOLDER_URL = "./data/countries";

    private Map<String, Long> popCountries; // save population of all countries
    private Map<String, Country> dataCountries; // save COVID19 data of all countries, utilize map for fast query

    public ReadDataFromCSV() {
        popCountries = new HashMap<>();
        dataCountries = new HashMap<>();
    }

    public Map<String, Country> getDataCountries() {
        readPopDataFromFile();
        readCOVIDDataFromFile();

        return dataCountries;
    }

    private void readCOVIDDataFromFile() {
        File folder = new File(DATA_COUNTRIES_FOLDER_URL);
        if (!folder.isDirectory())
            return;

        // read each file
        for (File file : Objects.requireNonNull(folder.listFiles())) {
            String fileName = file.getName();
            String countryName = fileName.split("\\.")[0].toLowerCase();

            // skip countries which we do not have population data
            if (popCountries.containsKey(countryName)) {
                Country country = new Country(countryName, popCountries.get(countryName), file);
                dataCountries.put(countryName, country);
            }
        }
    }

    private void printPopData(Map<String, Long> popCountries) {
        for(Map.Entry<String, Long> entry : popCountries.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
    }

    private void readPopDataFromFile() {
        try {
            File file = new File(POPULATION_FILE_URL);
            BufferedReader br = new BufferedReader(new FileReader(file));

            String line;
            String[] tempArr;
            while((line = br.readLine()) != null) {
                tempArr = line.split(",");
                String countryName = tempArr[0].toLowerCase();
                long pop = Long.parseLong(tempArr[1]);
                popCountries.put(countryName, pop);
            }

            br.close();
        } catch (Exception e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }
}
