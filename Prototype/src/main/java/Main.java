import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseEvent;
import javafx.stage.Stage;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Objects;

public class Main extends Application {
    @Override
    public void start(Stage stage) throws IOException {
    	Parent root = FXMLLoader.load(Objects.requireNonNull(getClass().getResource("/Checkout.fxml")));
        Scene scene = new Scene(root);
        stage.setTitle("Ticket Monarch");
        stage.setScene(scene);
        stage.show();
        
    	String fileName = "actionLog.txt";
    	File file = new File(fileName);
    	if(file.exists()) {
    		file.delete();
    	}
    	file.createNewFile();
        
        //Scuffed code to track events
        scene.addEventFilter(MouseEvent.MOUSE_MOVED, e -> {
            long ts = System.nanoTime()/1_000_000; // ms
            double x = e.getSceneX(), y = e.getSceneY();
            logEvent(ts, "mouse_move", x, y, e);
        });

        scene.addEventFilter(MouseEvent.MOUSE_PRESSED, e -> {
            long ts = System.nanoTime()/1_000_000;
            logEvent(ts, "mouse_down", e.getSceneX(), e.getSceneY(), e.getButton().toString());
        });

        scene.addEventFilter(KeyEvent.KEY_PRESSED, e -> {
            long ts = System.nanoTime()/1_000_000;
            logEvent(ts, "key_down", e.getCode().toString(), e.isShiftDown(), e.isControlDown());
        });
    }
    
    //Creates a string from the events
    private void logEvent(Object... args) {
        StringBuilder sb = new StringBuilder();
        for (Object arg : args) {
            sb.append(arg).append(" ");
        }
        //System.out.println(sb.toString()); //Printing to console for errors
        writeToFile(sb.toString());
    }
    
    //Writes event strings to file called actionLog.txt
    private void writeToFile (String line) {
    	File file = new File("actionLog.txt");
    	
        try {
            FileWriter fw = new FileWriter(file, true);
            BufferedWriter bw = new BufferedWriter(fw);

            bw.write(line);
            bw.newLine();
            
            bw.close();
        } catch (IOException e) {
            System.out.println("File writing error in " + file.getName());
        }
    }

    public static void main(String[] args) {
        launch();
    }
}
