// Java program for Fibonacci Series using Space
// Optimized Method
public class fibonacci {
	static int fib(int n)
	{
		int a = 0, b = 1, c;
		if (n == 0)
			return a;
		for (int i = 2; i <= n; i++) {
			c = a + b;
			a = b;
			b = c;
		}
		return b;
	}

	public static void main(String args[])
	{
		int n = 9;
		System.out.println(fib(n));
	}
};

// This code is contributed by Mihir Joshi

// Fibonacci Series using Recursion
import java.io.*;
class fibonacci {
	static int fib(int n)
	{
		if (n <= 1)
			return n;
		return fib(n - 1) + fib(n - 2);
	}

	public static void main(String args[])
	{
		int n = 9;
		System.out.println(
			n + "th Fibonacci Number: " + fib(n));
	}
}
/* This code is contributed by Rajat Mishra */

