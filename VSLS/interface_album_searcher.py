import cv2
import numpy as np
from typing import List, Optional, Tuple

from VSLS.interface_searcher import VSLSSearcher
from VSLS.interface_yolo import YoloInterface


class VSLSAlbumSearcher(VSLSSearcher):
    """
    Album Image Retrieval version của VSLSSearcher.

    Toàn bộ thuật toán VSLS (Adaptive Sampling, Gaussian/Spline/Uniform Distribution,
    Image Grid, YOLO Detection, Relation Scoring, Score Propagation, Target Management,
    Search Strategy) được KẾ THỪA NGUYÊN VẸN từ VSLSSearcher — không copy, không sửa
    logic thuật toán.

    Chỉ 2 điểm phụ thuộc vào "nguồn dữ liệu là video" được override:
        1. initialize_source(): thay vì mở video bằng cv2.VideoCapture, đặt
           raw_fps = fps = 1 và total_frame_num = số lượng ảnh trong album.
           Nhờ raw_fps == fps == 1, mọi công thức quy đổi chỉ số trong lớp cha
           (sample_frames, verify_and_remove_target, search, search_with_visualization)
           trở thành ánh xạ đồng nhất (index ảnh == index trong score_distribution),
           nên KHÔNG cần override thêm bất kỳ hàm nào khác.
        2. read_frame_batch(): đọc ảnh bằng cv2.imread() theo chỉ số trong
           self.image_paths, thay vì decord.VideoReader.

    Lưu ý về các tham số truyền ngoài (không hardcode trong logic):
        - search_budget: nên truyền 1.0 (quét toàn bộ album) vì mỗi album ML-CUFED
          chỉ có 30–100 ảnh, chi phí YOLO cho việc quét hết là không đáng kể, và
          quét hết giúp tránh bỏ sót ảnh chứa target/ground truth (khác với video
          dài hàng nghìn frame, nơi sampling một phần là cần thiết).
        - image_grid_shape: nên truyền đủ lớn (vd. (10, 10) = 100 ô) để một album
          lớn nhất (100 ảnh) lọt gọn vào đúng 1 grid, chỉ cần 1 lần gọi YOLO.
        - update_method: nên truyền "uniform" — phân phối lấy mẫu đều trên các ảnh
          CHƯA duyệt, không suy diễn điểm số theo khoảng cách chỉ số (vì thứ tự ảnh
          trong 1 album không mang ý nghĩa liên tục như timeline video).
      Tất cả các giá trị trên chỉ là DEFAULT của constructor bên dưới — người gọi
      hoàn toàn có thể truyền giá trị khác (vd. dùng lại "spline"/"gaussian" và
      search_budget nhỏ hơn) mà không phải sửa code của lớp này.
    """

    def __init__(
        self,
        image_paths: List[str],
        target_objects: List[str],
        cue_objects: List[str],
        relations: List[Tuple[str, str, str]],
        relation_alpha: float = 0.8,
        search_nframes: int = 8,
        image_grid_shape: Tuple[int, int] = (10, 10),
        search_budget: float = 1.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        confidence_threshold: float = 0.5,
        object2weight: Optional[dict] = None,
        yolo_scorer: Optional[YoloInterface] = None,
        update_method: str = "uniform",
    ):
        """
        Args:
            image_paths (List[str]): Danh sách đường dẫn ảnh trong album, vd.
                [".../001.jpg", ".../002.jpg", ...]. Thứ tự trong list quyết định
                chỉ số ảnh (image index) dùng xuyên suốt thuật toán.
            (Các tham số còn lại giữ nguyên ý nghĩa như VSLSSearcher.)
        """
        if not image_paths:
            raise ValueError("image_paths không được rỗng.")

        # QUAN TRỌNG: self.image_paths phải được gán TRƯỚC khi gọi super().__init__(),
        # vì VSLSSearcher.__init__() sẽ gọi self.initialize_source() (dispatch động
        # tới bản override bên dưới), và initialize_source() của lớp này cần
        # self.image_paths đã tồn tại để tính total_frame_num.
        self.image_paths = image_paths

        super().__init__(
            video_path=None,  # Album không dùng video_path; giữ tham số để tương thích
                               # chữ ký __init__ của lớp cha, KHÔNG được sử dụng ở đâu khác.
            target_objects=target_objects,
            cue_objects=cue_objects,
            relations=relations,
            relation_alpha=relation_alpha,
            search_nframes=search_nframes,
            image_grid_shape=image_grid_shape,
            search_budget=search_budget,
            output_dir=output_dir,
            prefix=prefix,
            confidence_threshold=confidence_threshold,
            object2weight=object2weight,
            yolo_scorer=yolo_scorer,
            update_method=update_method,
        )

    # ------------------------------------------------------------------ #
    # HOOK 1: khởi tạo nguồn dữ liệu
    # ------------------------------------------------------------------ #
    def initialize_source(self):
        """
        Override initialize_source() của VSLSSearcher.

        Album: mỗi ảnh là một đơn vị rời rạc, không có khái niệm fps/timeline như
        video. Đặt raw_fps = fps = 1 để mọi công thức quy đổi chỉ số trong lớp cha
        (frame_idx = int(sec * raw_fps / fps), timestamp = idx / fps) trở thành ánh
        xạ đồng nhất — image index == score_distribution index == timestamp index.

        Nhờ vậy, sample_frames(), verify_and_remove_target(), search(),
        search_with_visualization() không cần override.
        """
        self.raw_fps = 1
        self.fps = 1
        self.total_frame_num = len(self.image_paths)
        self.duration = self.total_frame_num

    # ------------------------------------------------------------------ #
    # HOOK 2: đọc dữ liệu ảnh theo chỉ số
    # ------------------------------------------------------------------ #
    def read_frame_batch(
        self, video_path: Optional[str], frame_indices: List[int]
    ) -> Tuple[List[int], List[np.ndarray]]:
        """
        Override read_frame_batch() của VSLSSearcher.

        Đọc ảnh bằng cv2.imread() theo chỉ số trong self.image_paths, thay vì
        decord.VideoReader. Chuyển BGR -> RGB để nhất quán định dạng màu với
        decord.VideoReader.get_batch() (vốn trả về RGB) mà các hàm dùng chung phía
        sau (create_image_grid, imageGridScoreFunction...) đang mong đợi.

        Args:
            video_path: KHÔNG dùng tới. Giữ lại để tương thích chữ ký với lớp cha,
                vì mọi lời gọi read_frame_batch() trong VSLSSearcher đều truyền
                self.video_path làm tham số đầu tiên.
            frame_indices (List[int]): Chỉ số ảnh trong self.image_paths cần đọc.

        Returns:
            Tuple[List[int], List[np.ndarray]]: Chỉ số ảnh và danh sách ảnh RGB
                tương ứng (dạng list, không stack thành ndarray vì các ảnh trong
                album có thể khác kích thước gốc — các nơi gọi hàm này đều resize
                ngay sau đó nên không yêu cầu shape đồng nhất tại bước đọc).
        """
        frames = []
        for idx in frame_indices:
            img_path = self.image_paths[idx]
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise ValueError(f"Không đọc được ảnh: {img_path}")
            frames.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        return frame_indices, frames


# Example usage
if __name__ == "__main__":
    import glob

    album_dir = "/path/to/album_001"
    image_paths = sorted(glob.glob(f"{album_dir}/*.jpg"))

    target_objects = ["person", "cake"]
    cue_objects = ["table", "candle"]
    relations = [
        ["person", "cake", "spatial"],
    ]

    searcher = VSLSAlbumSearcher(
        image_paths=image_paths,
        target_objects=target_objects,
        cue_objects=cue_objects,
        relations=relations,
        search_nframes=8,
        image_grid_shape=(10, 10),
        confidence_threshold=0.5,
        search_budget=1.0,
        update_method="uniform",
    )

    all_frames, time_stamps = searcher.search()
    print(f"Found {len(all_frames)} images, indices: {time_stamps}")
