
import java.util.Random;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;


public class RandomGenerator {
    public static double[][] normalDistribution(int rowCount, int colCount) {
        double mean = 0.0; // 平均
        double stdDev = 10.0; // 標準偏差
        long seed = 12345678;
        Random random = new Random(seed);
        double[][] randomNumbers = new double[rowCount][colCount];

        for (int i = 0; i < rowCount; i++) {
            for (int j = 0; j < colCount; j++) {
                double randomNumber = mean + stdDev * random.nextGaussian();
                randomNumbers[i][j] = randomNumber;
            }
        }
        return randomNumbers;
    }

    public static void execute() {


        int period = 14;
        int rowCount = 2 * period;
        int colCount = 10000;
        int colCountStart = 0;
        int colCountEnd = 10000;

        double[][] randomNumbers = normalDistribution(rowCount, colCount);

        for (int j = colCountStart; j < colCountEnd; j++) {
            for (int i = 0; i < rowCount/2; i++) {
              if (i == 8) {
                randomNumbers[2*i][j] = 0;
              } else {
              };
              if (i < 3) {
                randomNumbers[2*i+1][j] = 0;
              } else {
              }
            }
        }
        
        double[][] randomNumbersTransposed = new double[colCount][rowCount];
        for (int i = 0; i < rowCount; i++) {
            for (int j = 0; j < colCount; j++) {
                randomNumbersTransposed[j][i] = randomNumbers[i][j];
            }
        }

        // save from transport of randomNumbers to csv
        String fileName = "randomNumbers.csv";
        CsvFileWriter.write(fileName, randomNumbersTransposed);
        
        System.out.println("Done");
    }

    public static void main(String[] args) {
        execute();
    }
}

class CsvFileWriter {
    // CSVファイルにdouble配列を書き込むメソッド
    public static void write(String fileName, double[][] data) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            for (double[] row : data) {
                for (int i = 0; i < row.length; i++) {
                    writer.write(Double.toString(row[i]));
                    if (i < row.length - 1) {
                        writer.write(",");
                    }
                }
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}