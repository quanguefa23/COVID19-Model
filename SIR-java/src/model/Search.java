package model;

import java.util.*;

public class Search {

    public static final int SIZE = 1000000;
    public static final int SQUARE_N = 1000;
    public static int countBinarySearch = 0;
    public static int countInterpolationSearch = 0;

    public static void main(String... argv) {
        List<Integer> a = new ArrayList<>();
        Random r = new Random();
        for (int i = 0; i < SIZE; i++) {
            a.add(r.nextInt(SIZE));
        }
        int key = r.nextInt(SIZE);
        Collections.sort(a);

        System.out.println(binarySearch(a, 0, a.size() - 1, key));
        System.out.println(interpolationSearch(a, 0, a.size() - 1, key));
        System.out.println("Số thao tác Binary: " + countBinarySearch);
        System.out.println("Số thao tác Interpolation: " + countInterpolationSearch);
    }

    private static int binarySearch(List<Integer> a, int l, int r, int key) {
        countBinarySearch++;
        if (l > r)
            return -1;
        int mid = (l + r) / 2;
        if (key == a.get(mid))
            return mid;
        else if (key > a.get(mid))
            return binarySearch(a, mid + 1, r, key);
        else
            return binarySearch(a, l, mid - 1, key);
    }

    private static int interpolationSearch(List<Integer> a, int l, int r, int key) {
        countInterpolationSearch++;
        if (l > r || (l == r && a.get(l) != key) )
            return -1;
        else if (l == r && a.get(l) == key)
            return l;

        double p = 1.0 * (key - a.get(l)) / (a.get(r) - a.get(l));
        int mid = (int) (l + p * (r - l));

        int i = 1;
        if (key > a.get(mid)) {
            int next;
            while (true) {
                //countInterpolationSearch++;
                next = mid + i * SQUARE_N;
                if (next > r || key < a.get(next))
                    break;
                if (key == a.get(next))
                    return next;
                i++;
            }
            l = mid + (i - 1) * SQUARE_N + 1;
            r = Math.min(r, next - 1);
            return interpolationSearch(a, l, r, key);
        }
        else if (key < a.get(mid)) {
            int next;
            while (true) {
                //countInterpolationSearch++;
                next = mid - i * SQUARE_N;
                if (next < l || key > a.get(next))
                    break;
                if (key == a.get(next))
                    return next;
                i++;
            }
            r = mid - (i - 1) * SQUARE_N - 1;
            l = Math.min(l, next + 1);
            return interpolationSearch(a, l, r, key);
        }
        else
            return mid;
    }
}
