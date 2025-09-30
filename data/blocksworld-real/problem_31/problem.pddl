(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        orange - block
        red - block
        yellow - block
        green - block
        purple - block
        blue - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable orange)
        (on red orange)
        (on yellow red)
        (clear yellow)
        (ontable green)
        (clear green)
        (ontable purple)
        (clear purple)
        (ontable blue)
        (clear blue)
        ; orange -> red -> yellow
        ; green
        ; purple
        ; blue
    )
    (:goal (and (on purple yellow)
        (on red purple)
        (on blue green)
        (on orange blue)))
)
    