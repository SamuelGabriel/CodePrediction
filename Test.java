package hdfs;
import java.util.concurrent.*;

public class Test {
	int a = 0;
	public int add(int a, int b){
		a = a + ThreadLocalRandom.current().nextInt();
		return a+b;
	}
}
