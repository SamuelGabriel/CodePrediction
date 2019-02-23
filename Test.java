package hdfs;
import java.util.concurrent.ThreadLocalRandom;
import java.util.ArrayList;


public class Test {
	public String[] cars = {"Volvo", "BMW", "Ford", "Mazda"};
	public int plus_random_and_a(int b){
		ArrayList<Integer> c = new ArrayList<Integer>();
		c.add(4);
		b = 1;
		int d = 10;
		return cars[1].length() + c.get(0) + b + ThreadLocalRandom.current().nextInt();
	}
}
