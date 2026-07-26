"""
VSLSFrameworkAlbum: Album Image Retrieval version của VSLSFramework.

Kế thừa toàn bộ pipeline điều phối (Grounding -> Search -> Save -> QA) từ
VSLSFramework, chỉ override 3 hook phụ thuộc vào "nguồn dữ liệu là video":

    1. resolve_source_id()      -> định danh dùng để đặt tên thư mục output
    2. _grounder_source_kwargs() -> nguồn ảnh truyền cho VLM ở bước Grounding
    3. _create_searcher()        -> tạo VSLSAlbumSearcher thay vì VSLSSearcher

Theo yêu cầu: đưa TOÀN BỘ ảnh trong album (30-100 ảnh với ML-CUFED) vào cả
bước Grounding (interface_llm.VSLSUniversalGrounder.inference_query_grounding2)
lẫn bước Search (VSLSAlbumSearcher với search_budget=1.0, exhaustive).
"""

import os
import logging
from typing import List, Optional, Tuple

from VSLS.interface_llm import VSLSUniversalGrounder
from VSLS.interface_yolo import YoloInterface
from VSLS.interface_searcher import VSLSSearcher
from VSLS.interface_album_searcher import VSLSAlbumSearcher
from VSLS.VSLSFramework import VSLSFramework

logger = logging.getLogger(__name__)


class VSLSFrameworkAlbum(VSLSFramework):
    """
    Main class for performing object-based image search and question-answering
    trên MỘT ALBUM ảnh (thay vì 1 video).
    """

    def __init__(
        self,
        image_paths: List[str],
        yolo_scorer: YoloInterface,
        grounder: VSLSUniversalGrounder,
        question: str,
        options: str,
        search_nframes: int = 8,
        grid_rows: int = 10,
        grid_cols: int = 10,
        output_dir: str = './output',
        confidence_threshold: float = 0.6,
        search_budget: float = 1.0,
        prefix: str = 'stitched_image',
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda:7",
        update_method: str = "uniform",
        album_id: Optional[str] = None,
    ):
        """
        Args:
            image_paths (List[str]): Danh sách đường dẫn TOÀN BỘ ảnh trong album,
                vd. [".../001.jpg", ".../002.jpg", ...] (ML-CUFED: 30-100 ảnh/album).
            yolo_scorer, grounder, question, options, search_nframes, output_dir,
            confidence_threshold, prefix, config_path, checkpoint_path, device:
                giữ nguyên ý nghĩa như VSLSFramework.
            grid_rows, grid_cols (int, optional): Mặc định 10x10 = 100 ô, đủ chứa
                album lớn nhất (100 ảnh) trong 1 lần gọi YOLO. Vẫn là tham số
                truyền từ ngoài, có thể chỉnh nếu dataset khác có nhiều ảnh hơn.
            search_budget (float, optional): Mặc định 1.0 -- quét TOÀN BỘ ảnh
                trong album (khác với video, nơi 0.1 = 10% là đủ nhờ tính liên
                tục thời gian). Vẫn có thể truyền giá trị khác từ ngoài.
            update_method (str, optional): Mặc định "uniform" -- lấy mẫu đều trên
                các ảnh CHƯA duyệt, không suy diễn điểm theo khoảng cách chỉ số
                (xem VSLSAlbumSearcher / uniform_keyframe_distribution). Có thể
                truyền "spline"/"gaussian" nếu muốn hành vi giống video.
            album_id (Optional[str]): Định danh dùng để đặt tên thư mục output.
                Nếu không truyền, mặc định lấy tên thư mục cha của ảnh đầu tiên
                trong image_paths.
        """
        if not image_paths:
            raise ValueError("image_paths không được rỗng.")

        # QUAN TRỌNG: self.image_paths và self.album_id phải được gán TRƯỚC khi
        # gọi super().__init__(), vì VSLSFramework.__init__() sẽ gọi
        # self.resolve_source_id() (dispatch động tới bản override bên dưới),
        # và bản override đó cần self.image_paths / self.album_id đã tồn tại.
        self.image_paths = image_paths
        self.album_id = album_id

        super().__init__(
            video_path=None,  # Album không dùng video_path; giữ để tương thích
                               # chữ ký __init__ của lớp cha, KHÔNG dùng ở đâu khác.
            yolo_scorer=yolo_scorer,
            grounder=grounder,
            question=question,
            options=options,
            search_nframes=search_nframes,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            output_dir=output_dir,
            confidence_threshold=confidence_threshold,
            search_budget=search_budget,
            prefix=prefix,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            update_method=update_method,
        )

    # ------------------------------------------------------------------ #
    # HOOK 1: định danh nguồn dữ liệu (dùng đặt tên thư mục output)
    # ------------------------------------------------------------------ #
    def resolve_source_id(self) -> str:
        """
        Override resolve_source_id() của VSLSFramework.

        Ưu tiên dùng self.album_id nếu được truyền vào constructor; nếu không,
        mặc định lấy tên thư mục cha của ảnh đầu tiên trong image_paths (vd.
        image_paths = [".../album_0007/001.jpg", ...] -> source_id = "album_0007").
        """
        if self.album_id:
            return self.album_id
        return os.path.basename(os.path.dirname(self.image_paths[0]))

    # ------------------------------------------------------------------ #
    # HOOK 2: nguồn ảnh truyền cho Grounder (bước xác định target/cue objects)
    # ------------------------------------------------------------------ #
    def _grounder_source_kwargs(self) -> dict:
        """
        Override _grounder_source_kwargs() của VSLSFramework.

        Trả về TOÀN BỘ self.image_paths cho grounder.inference_query_grounding2()
        (thông qua tham số image_paths mới thêm ở interface_llm.py) -- tức là mọi
        ảnh trong album đều được đưa cho VLM ở bước Grounding, không sample như
        video, tránh bỏ sót object/ground truth nằm ở ảnh bị loại nếu chỉ lấy mẫu.
        """
        return {"video_path": None, "image_paths": self.image_paths}

    # ------------------------------------------------------------------ #
    # HOOK 3: khởi tạo searcher
    # ------------------------------------------------------------------ #
    def _create_searcher(self, target_objects, cue_objects, relations) -> VSLSSearcher:
        """
        Override _create_searcher() của VSLSFramework.

        Tạo VSLSAlbumSearcher (đọc ảnh bằng cv2.imread, exhaustive search) thay
        vì VSLSSearcher (đọc video bằng decord).
        """
        return VSLSAlbumSearcher(
            image_paths=self.image_paths,
            target_objects=target_objects,
            cue_objects=cue_objects,
            relations=relations,
            search_nframes=self.search_nframes,
            image_grid_shape=(self.grid_rows, self.grid_cols),
            output_dir=self.output_dir,
            confidence_threshold=self.confidence_threshold,
            search_budget=self.search_budget,
            yolo_scorer=self.yolo_scorer,
            update_method=self.update_method,
        )


# Example usage
if __name__ == "__main__":
    import glob

    album_dir = "/path/to/ML-CUFED/album_0007"
    image_paths = sorted(glob.glob(f"{album_dir}/*.jpg"))

    grounder = VSLSUniversalGrounder(
        backend="qwenvl",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        base_url="http://localhost:8000/v1",
    )

    # yolo_interface = ... (khởi tạo giống VSLSFramework.py)

    framework = VSLSFrameworkAlbum(
        image_paths=image_paths,
        yolo_scorer=None,  # thay bằng yolo_interface thật
        grounder=grounder,
        question="What is the color of the birthday cake?",
        options="A) Red\nB) White\nC) Blue\nD) Green",
        search_nframes=8,
        grid_rows=10,
        grid_cols=10,
        confidence_threshold=0.6,
        search_budget=1.0,
        update_method="uniform",
    )

    framework.run()
    print("Final Results:", framework.results)
