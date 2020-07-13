package entity;

public class DataEachDay {
    public long S; // susceptible
    public long I; // infected
    public long R; // recovered

    public DataEachDay(long N, long confirm, long death, long recover) {
        I = confirm - death - recover;
        R = death + recover;
        S = N - I - R;
    }

    public DataEachDay(long s, long i, long r) {
        S = s;
        I = i;
        R = r;
    }
}
