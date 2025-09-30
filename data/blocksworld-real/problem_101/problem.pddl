(define (problem blocksworld)
    (:domain blocksworld)
    (:objects
        blue - block
        green - block
        purple - block
        yellow - block
        red - block
        orange - block
        robot - robot
    )
    (:init
        (handempty robot)        (ontable blue)
        (on green blue)
        (clear green)
        (ontable purple)
        (on yellow purple)
        (clear yellow)
        (ontable red)
        (on orange red)
        (clear orange)
        ; blue -> green
        ; purple -> yellow
        ; red -> orange
    )
    (:goal (and (on yellow orange)
        (on red green)
        (on blue purple)))
)
    