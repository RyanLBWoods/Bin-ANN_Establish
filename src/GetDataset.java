import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class GetDataset {
    
    public static String[] getData(){
        
        try{
            FileReader fr = new FileReader("char.csv");
            BufferedReader br = new BufferedReader(fr);
            br.readLine();
            
            ArrayList<String> content = new ArrayList<String>();
            String column = null;
            while((column = br.readLine()) != null) {
                content.add(column);
            }
            
            String[] line = new String[content.size()];
            
            for (int i = 0; i < content.size(); i++){
                line[i] = content.get(i);
            }
            br.close();
            return line;
            
        } catch (FileNotFoundException e){
            System.out.println("File not found: " + e.getMessage());
            System.exit(0);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        return null;
    }
}
