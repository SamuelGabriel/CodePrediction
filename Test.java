package hdfs;
import java.util.concurrent.ThreadLocalRandom;


public class Test {
	public int a = 1;
	public int plus_random_and_a(int b){
		return a + b + ThreadLocalRandom.current().nextInt();
	}
}
