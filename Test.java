package hdfs;

public class Test {
	public int ret(int a){
		return java.util.concurrent.ThreadLocalRandom.current().nextInt() + a;
	}
	public int pa(int a){
		return ret(a) + ret(a);
	}
}
