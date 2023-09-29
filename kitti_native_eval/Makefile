all: main main_05
#main: evaluate_object_3d_offline.cpp
#	g++ -o evaluate_object_3d_offline2_recall  evaluate_object_3d_offline2_recall.cpp -lboost_system -lboost_filesystem

main: evaluate_object_3d_offline2.cpp
	g++ -o evaluate_object_3d_offline2  evaluate_object_3d_offline2.cpp -lboost_system -lboost_filesystem

main_05: evaluate_object_3d_offline_05_iou_new.cpp
	g++ -o evaluate_object_3d_offline_05_iou_new  evaluate_object_3d_offline_05_iou_new.cpp -lboost_system -lboost_filesystem

#main: evaluate_object_3d_offline_05_iou.cpp
#	g++ -o evaluate_object_3d_offline_05_iou_recall  evaluate_object_3d_offline_05_iou_recall.cpp -lboost_system -lboost_filesystem

#main: evaluate_object_3d_offline2_recall.cpp
#	g++ -o evaluate_object_3d_offline2_recall  evaluate_object_3d_offline2_recall.cpp -lboost_system -lboost_filesystem

.PHONY: clean

clean:
	rm -f evaluate_object_3d_offline
	rm -f evaluate_object_3d_offline_05_iou
